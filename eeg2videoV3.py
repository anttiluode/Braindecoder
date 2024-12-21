# =======================
# Imports
# =======================

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, hilbert
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib
import yaml
from datetime import datetime
import cv2

# Use 'Agg' backend for thread safety in matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =======================
# Logging Setup
# =======================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('eeg_image_decoder.log')
    ]
)
logger = logging.getLogger(__name__)

# =======================
# Configuration
# =======================

class Config:
    """Configuration handling for EEG and Image Processing"""
    def __init__(self, config_path=None):
        self.config = self.default_config()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._update_config(user_config)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def default_config(self):
        """Default configuration settings"""
        return {
            'eeg': {
                'gamma': {
                    'low_freq': 30.0,    # Gamma band lower cutoff (Hz)
                    'high_freq': 100.0,  # Gamma band upper cutoff (Hz)
                    'min_duration': 0.05,  # Minimum burst duration (s)
                    'max_duration': 0.15   # Maximum burst duration (s)
                },
                'alpha': {
                    'low_freq': 8.0,    # Alpha band lower cutoff (Hz)
                    'high_freq': 12.0,  # Alpha band upper cutoff (Hz)
                },
                'beta': {
                    'low_freq': 13.0,    # Beta band lower cutoff (Hz)
                    'high_freq': 30.0,   # Beta band upper cutoff (Hz)
                },
                'sampling': {
                    'original': 1200,     # Original sampling rate (Hz)
                    'target': 120         # Downsampled rate (Hz)
                },
                'window': {
                    'size': 0.5,          # Window size in seconds
                    'stride': 0.25        # Stride in seconds
                },
                'event_detection': {
                    'threshold': 2.0,     # Threshold for burst detection (std above mean)
                    'min_separation': 0.5 # Minimum separation between bursts (s)
                }
            },
            'clip': {
                'model_name': 'ViT-B/32',  # CLIP model variant
                'pretrained': True
            },
            'encoder': {
                'hidden_dims': [128, 64],
                'latent_dim': 512,  # CLIP embedding size
                'epochs': 20,
                'batch_size': 128,
                'learning_rate': 1e-3
            },
            'contrastive': {
                'temperature': 0.07,  # Temperature parameter for contrastive loss
                'lambda_contrastive': 0.5,  # Weight for contrastive loss
                'lambda_regression': 0.5    # Weight for regression loss
            },
            'dataset': {
                'cifar_root': './data/cifar10',
                'download': True
            },
            'video': {
                'fps': 24,
                'resolution': [256, 256],
                'output_path': 'eeg_reconstruction.mp4',
                'gamma_bursts_video_path': 'gamma_bursts.mp4'
            }
        }

    def _update_config(self, user_config):
        """Recursively updates the default config with user config"""
        for key, value in user_config.items():
            if key in self.config and isinstance(self.config[key], dict):
                self._update_config_recursive(self.config[key], value)
            else:
                self.config[key] = value

    def _update_config_recursive(self, base, updates):
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict):
                self._update_config_recursive(base[key], value)
            else:
                base[key] = value

    def __getitem__(self, key):
        """Allow subscript access to the config dictionary."""
        return self.config[key]

# =======================
# Project Paths
# =======================

class ProjectPaths:
    """Manages project directories and paths"""
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent
        self.base_dir = Path(base_dir)
        
        # Define directories
        self.models_dir = self.base_dir / 'models'
        self.data_dir = self.base_dir / 'data'
        self.results_dir = self.base_dir / 'results'
        self.eeg_dir = self.data_dir / 'eeg'
        self.processed_dir = self.data_dir / 'processed'
        self.plots_dir = self.results_dir / 'plots'
        self.generated_images_dir = self.results_dir / 'generated_images'
        
        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.data_dir, self.results_dir, 
                        self.eeg_dir, self.processed_dir, self.plots_dir, self.generated_images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name):
        return self.models_dir / f"{model_name}.pth"
    
    def get_plot_path(self, base_name, plot_type):
        """Get path for saving plots"""
        return self.plots_dir / f"{base_name}_{plot_type}.png"
    
    def get_processed_path(self, base_name):
        """Get directory for processed data"""
        return self.processed_dir / base_name
    
    def get_generated_images_path(self, save_dir='generated_images'):
        return self.generated_images_dir / save_dir

# =======================
# EEG Processing
# =======================

class EEGProcessor:
    """Processes EEG data for gamma burst detection and window extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.eeg_config = config.config['eeg']
        self.raw = None
        self.sampling_rate = None
        self.channel_names = None
        self.selected_channel = None
        self.scaler = StandardScaler()
    
    def load_eeg(self, edf_path, selected_channel=None):
        """Load EEG file and select channel"""
        logger.info(f"Loading EEG data from {edf_path}")
        self.raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        self.sampling_rate = self.raw.info['sfreq']
        self.channel_names = self.raw.ch_names
        
        if selected_channel:
            if selected_channel in self.channel_names:
                self.selected_channel = selected_channel
            else:
                raise ValueError(f"Channel {selected_channel} not found in EEG data.")
        else:
            # Default to a posterior channel if not specified
            posterior_channels = [ch for ch in self.channel_names if 'Pz' in ch or 'Oz' in ch]
            if posterior_channels:
                self.selected_channel = posterior_channels[0]
                logger.info(f"Defaulting to posterior channel: {self.selected_channel}")
            else:
                self.selected_channel = self.channel_names[0]
                logger.info(f"No posterior channel found. Defaulting to first channel: {self.selected_channel}")
        
        logger.info(f"Selected channel for processing: {self.selected_channel}")
        return self.raw
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass Butterworth filter"""
        nyq = 0.5 * self.sampling_rate
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = butter(N=4, Wn=[low, high], btype='band')
        filtered = filtfilt(b, a, data)
        return filtered
    
    def detect_bursts(self, data, band):
        """Detect bursts in the specified frequency band"""
        logger.info(f"Detecting {band} bursts...")
        low_freq = self.eeg_config[band]['low_freq']
        high_freq = self.eeg_config[band]['high_freq']
        filtered = self.apply_bandpass_filter(data, low_freq, high_freq)
        analytic = hilbert(filtered)
        power = np.abs(analytic) ** 2
        
        # Smooth power with 50ms window
        smooth_window = int(0.05 * self.eeg_config['sampling']['target'])
        power_smooth = np.convolve(power, np.ones(smooth_window)/smooth_window, mode='same')
        
        # Threshold for burst detection
        threshold = np.mean(power_smooth) + self.eeg_config['event_detection']['threshold'] * np.std(power_smooth)
        above_threshold = power_smooth > threshold
        
        # Find burst starts and ends
        burst_starts = np.where(np.diff(above_threshold.astype(int)) > 0)[0]
        burst_ends = np.where(np.diff(above_threshold.astype(int)) < 0)[0]
        
        if len(burst_starts) == 0 or len(burst_ends) == 0:
            logger.warning(f"No {band} bursts detected.")
            return []
        
        # Ensure equal number of starts and ends
        if burst_starts[0] > burst_ends[0]:
            burst_ends = burst_ends[1:]
        if len(burst_starts) > len(burst_ends):
            burst_starts = burst_starts[:len(burst_ends)]
        
        # Filter bursts by duration and separation
        min_duration = self.eeg_config['gamma']['min_duration'] * self.eeg_config['sampling']['target']
        max_duration = self.eeg_config['gamma']['max_duration'] * self.eeg_config['sampling']['target']
        min_separation = self.eeg_config['event_detection']['min_separation'] * self.eeg_config['sampling']['target']
        
        bursts = []
        last_end = -min_separation
        for start, end in zip(burst_starts, burst_ends):
            duration = end - start
            if min_duration <= duration <= max_duration:
                peak = start + np.argmax(power_smooth[start:end])
                if (peak - last_end) >= min_separation:
                    bursts.append({
                        'band': band,
                        'start': start,
                        'peak': peak,
                        'end': end,
                        'duration': duration / self.eeg_config['sampling']['target'],
                        'power': power_smooth[peak]
                    })
                    last_end = end
        logger.info(f"Detected {len(bursts)} {band} bursts.")
        return bursts
    
    def extract_windows(self, bursts):
        """Extract short windows around detected bursts"""
        logger.info("Extracting windows around detected bursts...")
        window_size = self.eeg_config['window']['size']  # seconds
        window_samples = int(window_size * self.eeg_config['sampling']['target'])
        
        data = self.raw.get_data(picks=self.selected_channel)[0]
        downsample_factor = int(self.sampling_rate / self.eeg_config['sampling']['target'])
        downsampled = data[::downsample_factor]
        
        windows = []
        for burst in bursts:
            peak = burst['peak']
            # Extract window centered around the peak
            start = peak - window_samples // 2
            end = peak + window_samples // 2
            if start < 0 or end > len(downsampled):
                continue
            window = downsampled[start:end]
            window = self.scaler.fit_transform(window.reshape(-1, 1)).ravel()
            window = np.clip(window, -20, 20)  # Clip extreme values
            windows.append(window)
        
        logger.info(f"Extracted {len(windows)} windows from all frequency bands.")
        return np.array(windows)

# =======================
# Dataset
# =======================

class EEGImageDataset(Dataset):
    """Dataset combining EEG windows and CLIP image embeddings"""
    def __init__(self, eeg_windows, image_latents, transform=None):
        self.eeg_windows = eeg_windows
        self.image_latents = image_latents
        self.transform = transform
        
        assert len(self.eeg_windows) == len(self.image_latents), "Mismatch between EEG windows and image latents."
        logger.info(f"EEG-Image Dataset: {len(self.eeg_windows)} samples.")
        if self.transform is None:
            logger.warning("No transform applied to images.")
        else:
            logger.info("Transform applied to images.")
    
    def __len__(self):
        return len(self.eeg_windows)
    
    def __getitem__(self, idx):
        eeg = self.eeg_windows[idx]
        z = self.image_latents[idx]
        
        if self.transform:
            try:
                # Assuming that image_latents are precomputed and correspond to images
                # If images are needed for transformations, additional adjustments are required
                pass  # Placeholder: Adjust based on how image_latents are obtained
            except Exception as e:
                logger.error(f"Transform failed for image at index {idx}: {e}")
                raise e
        
        # Debugging: Check if z is a tensor
        if not isinstance(z, torch.Tensor):
            logger.error(f"Image latent at index {idx} is not a tensor.")
            raise TypeError(f"Image latent at index {idx} is not a tensor.")
        else:
            logger.debug(f"Image latent at index {idx} is a tensor with shape {z.shape}")
        
        return {
            'eeg': torch.tensor(eeg, dtype=torch.float32),
            'image_latent': z
        }

# =======================
# Models
# =======================

class EEGEncoder(nn.Module):
    """Enhanced Encoder that maps EEG windows to CLIP latent space using CNN layers"""
    def __init__(self, input_dim, hidden_dims=[128, 64], latent_dim=512):
        super(EEGEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        conv_output_dim = (input_dim // 2) // 2 * 64  # Assuming two pooling layers
        
        self.fc1 = nn.Linear(conv_output_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], latent_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# =======================
# Loss Functions
# =======================

class ContrastiveLoss(nn.Module):
    """Contrastive loss (InfoNCE)"""
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, z_eeg, z_image):
        """
        z_eeg: Tensor of shape (N, D)
        z_image: Tensor of shape (N, D)
        """
        batch_size = z_eeg.size(0)
        z_eeg = F.normalize(z_eeg, dim=1)
        z_image = F.normalize(z_image, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_eeg, z_image.T) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size).to(z_eeg.device)
        
        # Loss for EEG to Image
        loss_e2i = F.cross_entropy(similarity_matrix, labels)
        # Loss for Image to EEG
        loss_i2e = F.cross_entropy(similarity_matrix.T, labels)
        
        loss = (loss_e2i + loss_i2e) / 2
        return loss

# =======================
# Training Functions
# =======================

def train_eeg_encoder(model, dataloader, optimizer, device, clip_model, contrastive_loss_fn, config, epochs=20):
    """Train EEG Encoder to map EEG to CLIP latent space with both contrastive and regression losses"""
    model.train()
    clip_model.eval()
    mse_loss_fn = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_contrastive_loss = 0
        total_regression_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"EEG Encoder Epoch {epoch}")):
            try:
                eeg = batch['eeg'].to(device).float()  # Ensure float32
                z_image = batch['image_latent'].to(device).float()  # Ensure float32
                
                # Verify dtypes
                assert eeg.dtype == torch.float32, f"EEG dtype mismatch: {eeg.dtype}"
                assert z_image.dtype == torch.float32, f"Image latent dtype mismatch: {z_image.dtype}"
                
                optimizer.zero_grad()
                z_eeg = model(eeg)
                
                # Verify z_eeg dtype
                assert z_eeg.dtype == torch.float32, f"z_eeg dtype mismatch: {z_eeg.dtype}"
                
                # Contrastive Loss
                contrastive_loss = contrastive_loss_fn(z_eeg, z_image)
                
                # Regression Loss
                regression_loss = mse_loss_fn(z_eeg, z_image)
                
                # Combined Loss
                combined_loss = (config.config['contrastive']['lambda_contrastive'] * contrastive_loss +
                                 config.config['contrastive']['lambda_regression'] * regression_loss)
                
                combined_loss.backward()
                optimizer.step()
                
                total_loss += combined_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_regression_loss += regression_loss.item()
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} in epoch {epoch}: {e}")
                raise e  # Re-raise the exception after logging
        
        avg_loss = total_loss / len(dataloader.dataset)
        avg_contrastive_loss = total_contrastive_loss / len(dataloader.dataset)
        avg_regression_loss = total_regression_loss / len(dataloader.dataset)
        logger.info(f"EEG Encoder Epoch {epoch}, Combined Loss: {avg_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}, Regression Loss: {avg_regression_loss:.4f}")
    return model

# =======================
# Evaluation Metrics
# =======================

from skimage.metrics import structural_similarity as ssim

def compute_ssim_metric(generated, original):
    """Compute Structural Similarity Index Measure between two images"""
    ssim_score, _ = ssim(generated, original, full=True, multichannel=True)
    return ssim_score

def compute_pixcorr_metric(generated, original):
    """Compute Pixel-wise Correlation between two images"""
    generated_flat = generated.flatten()
    original_flat = original.flatten()
    correlation = np.corrcoef(generated_flat, original_flat)[0, 1]
    return correlation

# =======================
# GUI Implementation
# =======================

import clip
from PIL import Image

class EEGImageDecoderGUI:
    """GUI for EEG Image Decoding Project"""
    
    def __init__(self, config: Config, paths: ProjectPaths):
        self.config = config
        self.paths = paths
        self.processor = EEGProcessor(config)
        self.eeg_encoder = EEGEncoder(
            input_dim=int(self.config.config['eeg']['window']['size'] * self.config.config['eeg']['sampling']['target']),
            hidden_dims=self.config.config['encoder']['hidden_dims'],
            latent_dim=self.config.config['encoder']['latent_dim']
        ).to(self.config.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
        self.dataset = None
        self.dataloader = None
        self.encoder_trained = False
        self.generated_images = None  # To store generated images
        
        # Load CLIP model
        logger.info("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load(
            self.config.config['clip']['model_name'],
            device=self.config.device,
            jit=False  # Disable JIT to prevent using half precision
        )
        self.clip_model = self.clip_model.float()  # Convert CLIP model to float32
        self.clip_model.eval()
        logger.info("CLIP model loaded in float32.")
        
        # Initialize GUI components
        self.root = tk.Tk()
        self.root.title("EEG Image Decoder")
        self.root.geometry("1200x800")
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup main GUI components"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="EEG File Selection")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=80)
        file_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        load_btn = ttk.Button(file_frame, text="Load EEG", command=self.load_eeg)
        load_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Channel selection frame
        channel_frame = ttk.LabelFrame(self.root, text="Select EEG Channel")
        channel_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.channel_listbox = tk.Listbox(channel_frame, height=5, selectmode=tk.SINGLE)
        self.channel_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        channel_scrollbar = ttk.Scrollbar(channel_frame, orient=tk.VERTICAL, command=self.channel_listbox.yview)
        channel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.channel_listbox.config(yscrollcommand=channel_scrollbar.set)
        
        # Processing controls frame
        process_frame = ttk.LabelFrame(self.root, text="Processing Controls")
        process_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Epoch input for EEG Encoder
        ttk.Label(process_frame, text="EEG Encoder Epochs:").pack(side=tk.LEFT, padx=5, pady=5)
        self.encoder_epochs_var = tk.IntVar(value=self.config.config['encoder']['epochs'])
        encoder_epochs_entry = ttk.Entry(process_frame, textvariable=self.encoder_epochs_var, width=5)
        encoder_epochs_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        train_encoder_btn = ttk.Button(process_frame, text="Train EEG Encoder", command=self.train_eeg_encoder)
        train_encoder_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        process_eeg_btn = ttk.Button(process_frame, text="Process EEG Data", command=self.process_eeg_data)
        process_eeg_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        generate_images_btn = ttk.Button(process_frame, text="Generate Images", command=self.generate_images)
        generate_images_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        generate_video_btn = ttk.Button(process_frame, text="Generate Video", command=self.generate_video)
        generate_video_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Additional controls for visualizations
        visualize_frame = ttk.LabelFrame(self.root, text="Visualizations")
        visualize_frame.pack(fill=tk.X, padx=10, pady=5)
        
        generate_gamma_video_btn = ttk.Button(visualize_frame, text="Generate Gamma Bursts Video", command=self.generate_gamma_video)
        generate_gamma_video_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        generate_bursts_plot_btn = ttk.Button(visualize_frame, text="Generate Bursts Location Plot", command=self.generate_bursts_plot)
        generate_bursts_plot_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status and logs
        status_frame = ttk.LabelFrame(self.root, text="Status and Logs")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(status_frame, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Redirect logger to GUI
        self.redirect_logs()
    
    def redirect_logs(self):
        """Redirect logs to the GUI text widget"""
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.configure(state='disabled')
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, append)
        
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(text_handler)
    
    def browse_file(self):
        """Browse and select an EEG EDF file"""
        file_path = filedialog.askopenfilename(
            initialdir=str(self.paths.eeg_dir),
            title="Select EEG EDF File",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def load_eeg(self):
        """Load EEG data and populate channel list"""
        try:
            file_path = self.file_path_var.get()
            if not file_path:
                raise ValueError("Please select an EEG file.")
            # Optionally, allow the user to select a channel after loading
            self.processor.load_eeg(file_path)
            self.populate_channels()
            messagebox.showinfo("Success", "EEG data loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logger.error(f"Error loading EEG: {str(e)}")
    
    def populate_channels(self):
        """Populate the channel listbox"""
        self.channel_listbox.delete(0, tk.END)
        for ch in self.processor.channel_names:
            self.channel_listbox.insert(tk.END, ch)
        # Select the first channel by default
        if self.channel_listbox.size() > 0:
            self.channel_listbox.select_set(0)
            self.channel_var = self.channel_listbox.get(0)
            logger.info(f"Default channel selected: {self.channel_var}")
    
    def check_model_exists(self, model_path):
        """Check if a model file exists"""
        return model_path.exists()
    
    def prompt_overwrite(self, model_name):
        """Prompt the user to overwrite an existing model"""
        response = messagebox.askyesno(
            "Model Exists",
            f"The model '{model_name}' already exists. Do you want to overwrite it?"
        )
        return response
    
    def train_eeg_encoder(self):
        """Train the EEG Encoder to map EEG to CLIP latent space"""
        try:
            encoder_path = self.paths.get_model_path('eeg_encoder')
            if self.check_model_exists(encoder_path):
                if not self.prompt_overwrite('eeg_encoder'):
                    logger.info("EEG Encoder training skipped by user.")
                    return
            logger.info("Starting EEG Encoder training...")
            # Assuming you have paired EEG windows and image latents
            if self.dataset is None:
                messagebox.showerror("Error", "Please process EEG data first.")
                return
            # Create DataLoader
            train_loader = DataLoader(
                self.dataset, 
                batch_size=self.config.config['encoder']['batch_size'],
                shuffle=True, 
                num_workers=0  # Set num_workers=0 for debugging
            )
            # Initialize optimizer
            optimizer = torch.optim.Adam(
                self.eeg_encoder.parameters(), 
                lr=self.config.config['encoder']['learning_rate']
            )
            # Initialize Contrastive Loss
            contrastive_loss_fn = ContrastiveLoss(temperature=self.config.config['contrastive']['temperature'])
            # Train EEG Encoder
            self.eeg_encoder = train_eeg_encoder(
                self.eeg_encoder, 
                train_loader, 
                optimizer, 
                self.config.device, 
                self.clip_model, 
                contrastive_loss_fn,
                self.config,
                epochs=self.encoder_epochs_var.get()
            )
            self.encoder_trained = True
            # Save the trained EEG Encoder
            torch.save(self.eeg_encoder.state_dict(), encoder_path)
            logger.info(f"EEG Encoder trained and saved to {encoder_path}")
            messagebox.showinfo("Success", "EEG Encoder training completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during EEG Encoder training: {str(e)}")
            logger.error(f"Error during EEG Encoder training: {str(e)}")
    
    def process_eeg_data(self):
        """Detect bursts and extract windows"""
        try:
            if not hasattr(self, 'processor') or self.processor.raw is None:
                raise ValueError("Please load EEG data first.")
            data = self.processor.raw.get_data(picks=self.processor.selected_channel)[0]
            # Detect bursts across multiple frequency bands
            bursts_alpha = self.processor.detect_bursts(data, 'alpha')
            bursts_beta = self.processor.detect_bursts(data, 'beta')
            bursts_gamma = self.processor.detect_bursts(data, 'gamma')
            all_bursts = bursts_alpha + bursts_beta + bursts_gamma
            windows = self.processor.extract_windows(all_bursts)
            
            if len(windows) == 0:
                raise ValueError("No valid EEG windows extracted.")
            
            # Assign corresponding image latents
            # For meaningful alignment, ensure that each EEG window is paired with the correct image latent
            # This requires that you have the image corresponding to each EEG window
            # Here, we'll assume that the order of windows corresponds to the order of images
            cifar_images = self.load_cifar_images(len(windows))
            if len(windows) != len(cifar_images):
                min_len = min(len(windows), len(cifar_images))
                windows = windows[:min_len]
                cifar_images = cifar_images[:min_len]
            
            # Encode images to obtain their CLIP latent representations
            image_latents = self.encode_images(cifar_images)
            
            # Create dataset and dataloader
            self.dataset = EEGImageDataset(
                eeg_windows=windows, 
                image_latents=image_latents, 
                transform=self.transform
            )
            self.dataloader = DataLoader(
                self.dataset, 
                batch_size=self.config.config['encoder']['batch_size'],
                shuffle=True, 
                num_workers=0  # Set num_workers=0 for debugging
            )
            
            # Test the DataLoader by fetching one batch
            try:
                batch = next(iter(self.dataloader))
                eeg_batch = batch['eeg']
                z_batch = batch['image_latent']
                logger.info(f"Sample Batch - EEG Shape: {eeg_batch.shape}, Image Latent Shape: {z_batch.shape}")
            except Exception as e:
                logger.error(f"Error fetching a batch from DataLoader: {e}")
                raise e
            
            logger.info("Dataset and DataLoader initialized.")
            messagebox.showinfo("Success", "EEG data processed and windows extracted successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing EEG data: {str(e)}")
            logger.error(f"Error processing EEG data: {str(e)}")
    
    def load_cifar_images(self, num_images):
        """Load CIFAR-10 images and select a subset directly without using DataLoader."""
        cifar_dataset = datasets.CIFAR10(
            root=self.config.config['dataset']['cifar_root'],
            train=True,
            download=self.config.config['dataset']['download'],
            transform=self.transform  # Apply transforms here if needed
        )
        
        # Randomly select indices
        indices = np.random.choice(len(cifar_dataset), num_images, replace=False)
        images = [cifar_dataset[i][0] for i in indices]
        
        # Verify that all images are tensors
        for idx, img in enumerate(images):
            if not isinstance(img, torch.Tensor):
                logger.error(f"CIFAR-10 image at index {idx} is not a torch.Tensor.")
                raise TypeError(f"CIFAR-10 image at index {idx} is not a torch.Tensor.")
        
        logger.info(f"Loaded {len(images)} CIFAR-10 images as torch.Tensor objects.")
        return images
    
    def encode_images(self, images):
        """Encode images using the pretrained CLIP model to obtain latent representations."""
        if not images:
            raise ValueError("No images provided for encoding.")
        self.clip_model.eval()
        with torch.no_grad():
            image_latents = []
            for img in images:
                # Ensure image is in PIL format
                if not isinstance(img, Image.Image):
                    img_pil = transforms.ToPILImage()(img)
                else:
                    img_pil = img
                img_preprocessed = self.clip_preprocess(img_pil).unsqueeze(0).to(self.config.device)
                z = self.clip_model.encode_image(img_preprocessed)
                z = F.normalize(z, dim=-1).float()  # Ensure float32
                image_latents.append(z.squeeze(0).cpu())
            image_latents = torch.stack(image_latents)
        logger.info(f"Encoded images into CLIP latent space with shape: {image_latents.shape}")
        return image_latents
    
    def generate_images(self):
        """Generate images from EEG data"""
        if not self.encoder_trained:
            messagebox.showerror("Error", "Please train the EEG Encoder first.")
            return
        try:
            logger.info("Generating images from EEG data...")
            # Process windows through EEG Encoder to get latent vectors
            windows = self.dataset.eeg_windows
            windows_tensor = torch.FloatTensor(windows).to(self.config.device)
            with torch.no_grad():
                z_eeg = self.eeg_encoder(windows_tensor)
                z_eeg = F.normalize(z_eeg, dim=1)  # Normalize for better decoding
                # Here, you need a generative model that can generate images from CLIP embeddings
                # For demonstration, we'll assume you have a function `generate_image_from_clip` 
                # which is a placeholder
                generated_images = self.generate_image_from_clip(z_eeg)
                generated_images = generated_images.cpu().numpy().transpose(0, 2, 3, 1)
            
            # Debugging: Log the shape and data type
            logger.debug(f"Generated Images Shape: {generated_images.shape}")
            logger.debug(f"Generated Images Data Type: {generated_images.dtype}")
            
            # Save generated images
            self.save_generated_images(generated_images, save_dir='generated_images')
            
            # Store generated images for video generation
            self.generated_images = generated_images
            
            # Display images in the GUI
            self.display_images(generated_images)
            logger.info("Image generation completed.")
            messagebox.showinfo("Success", "Images generated and saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating images: {str(e)}")
            logger.error(f"Error generating images: {str(e)}")
    
    def generate_image_from_clip(self, z_clip):
        """
        Placeholder function for generating images from CLIP embeddings.
        You need to implement or integrate a generative model (e.g., diffusion model) that can
        generate images conditioned on CLIP embeddings.
        
        For demonstration purposes, we'll return random noise images.
        """
        batch_size = z_clip.size(0)
        generated = torch.randn(batch_size, 3, 224, 224).to(self.config.device)
        generated = torch.tanh(generated)  # Scale to [-1, 1]
        return generated
    
    def save_generated_images(self, images, save_dir='generated_images'):
        """
        Save generated images to the specified directory.

        Parameters:
        - images (numpy.ndarray): Array of generated images with shape (N, H, W, C) and values in [-1, 1].
        - save_dir (str): Directory where images will be saved.
        """
        try:
            # Ensure the save directory exists
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving generated images to '{save_dir}'...")
            
            for idx, img in enumerate(images):
                # Convert image from [-1, 1] to [0, 1], then to [0, 255]
                img_uint8 = ((img + 1) / 2 * 255).astype(np.uint8)  # Assuming images are in [-1, 1]
                
                # Convert numpy array to PIL Image
                img_pil = Image.fromarray(img_uint8)
                
                # Define the file path with leading zeros for better sorting
                img_path = os.path.join(save_dir, f"image_{idx+1:04d}.png")
                
                # Save the image
                img_pil.save(img_path)
            
            logger.info(f"Saved {len(images)} generated images to '{save_dir}'.")
        except Exception as e:
            logger.error(f"Failed to save generated images: {e}")
            messagebox.showerror("Error", f"Failed to save generated images: {e}")
    
    def display_images(self, images):
        """Display generated images in grid"""
        if images is None or len(images) == 0:
            logger.warning("No images to display.")
            return
        # Create a new window to display images
        display_window = tk.Toplevel(self.root)
        display_window.title("Generated Images")
        display_window.geometry("800x600")
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(display_window)
        scrollbar = ttk.Scrollbar(display_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Define grid size
        columns = 4
        for i, img in enumerate(images):
            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=i//columns, column=i%columns, padx=5, pady=5)
            
            # Convert to PhotoImage
            img_pil = Image.fromarray(((img + 1) / 2 * 255).astype(np.uint8))  # Assuming images are in [-1,1]
            img_pil = img_pil.resize((100, 100))
            img_tk = ImageTk.PhotoImage(img_pil)
            
            label = ttk.Label(frame, image=img_tk)
            label.image = img_tk  # Keep a reference
            label.pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def generate_video(self):
        """Generate video from generated images"""
        if not self.encoder_trained:
            messagebox.showerror("Error", "Please train the EEG Encoder first.")
            return
        try:
            if self.generated_images is None or len(self.generated_images) == 0:
                raise ValueError("Please generate images first.")
            logger.info("Starting video generation...")
            video_path = self.config.config['video']['output_path']
            fps = self.config.config['video']['fps']
            resolution = tuple(self.config.config['video']['resolution'])
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, resolution)
            
            # Add images to video
            for img in self.generated_images:
                # Convert from RGB to BGR
                img_bgr = cv2.cvtColor(((img + 1) / 2 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                # Resize to desired resolution
                img_resized = cv2.resize(img_bgr, resolution)
                video.write(img_resized)
            
            video.release()
            logger.info(f"Video saved to {video_path}")
            messagebox.showinfo("Success", f"Video generated and saved to {video_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating video: {str(e)}")
            logger.error(f"Error generating video: {str(e)}")
    
    # Additional controls for visualizations
    def generate_gamma_video(self):
        """Generate a video of the gamma bursts only"""
        try:
            if not hasattr(self, 'processor') or self.processor.raw is None:
                raise ValueError("Please load and process EEG data first.")
            data = self.processor.raw.get_data(picks=self.processor.selected_channel)[0]
            bursts = self.processor.detect_bursts(data, 'gamma')
            if not bursts:
                raise ValueError("No gamma bursts detected to generate video.")
            
            logger.info("Generating gamma bursts video...")
            video_path = self.config.config['video']['gamma_bursts_video_path']
            fps = self.config.config['video']['fps']
            resolution = tuple(self.config.config['video']['resolution'])
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, resolution)
            
            for burst in bursts:
                start = burst['start']
                end = burst['end']
                # Extract the burst window from raw data
                burst_data = self.processor.raw.get_data(picks=self.processor.selected_channel)[0][start:end]
                # Normalize for visualization
                burst_data = (burst_data - np.min(burst_data)) / (np.max(burst_data) - np.min(burst_data))
                # Convert to grayscale image
                burst_image = np.tile(burst_data, (256, 256, 1))
                burst_image = (burst_image * 255).astype(np.uint8)
                burst_image = cv2.cvtColor(burst_image, cv2.COLOR_GRAY2BGR)
                video.write(burst_image)
            
            video.release()
            logger.info(f"Gamma bursts video saved to {video_path}")
            messagebox.showinfo("Success", f"Gamma bursts video generated and saved to {video_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating gamma bursts video: {str(e)}")
            logger.error(f"Error generating gamma bursts video: {str(e)}")
    
    def generate_bursts_plot(self):
        """Generate an image showing where bursts were found in the data"""
        try:
            if not hasattr(self, 'processor') or self.processor.raw is None:
                raise ValueError("Please load and process EEG data first.")
            data = self.processor.raw.get_data(picks=self.processor.selected_channel)[0]
            bursts = self.processor.detect_bursts(data, 'gamma')
            if not bursts:
                raise ValueError("No gamma bursts detected to plot.")
            
            logger.info("Generating bursts location plot...")
            fig, ax = plt.subplots(figsize=(10, 4))
            times = np.arange(len(data)) / self.processor.sampling_rate
            ax.plot(times, data, label='EEG Signal')
            
            for burst in bursts:
                ax.axvspan(burst['start']/self.processor.sampling_rate, 
                          burst['end']/self.processor.sampling_rate, 
                          color='red', alpha=0.3)
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Gamma Bursts in Channel: {self.processor.selected_channel}")
            ax.legend()
            
            plot_path = self.paths.get_plot_path('gamma_bursts', 'location')
            plt.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Bursts location plot saved to {plot_path}")
            
            # Display the plot in a new window
            self.display_plot(plot_path)
            messagebox.showinfo("Success", f"Bursts location plot generated and saved to {plot_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating bursts location plot: {str(e)}")
            logger.error(f"Error generating bursts location plot: {str(e)}")
    
    def display_plot(self, plot_path):
        """Display the generated plot in a new window"""
        try:
            plot_image = Image.open(plot_path)
            plot_image = plot_image.resize((800, 400))
            plot_tk = ImageTk.PhotoImage(plot_image)
            
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Bursts Location Plot")
            plot_window.geometry("820x420")
            
            label = ttk.Label(plot_window, image=plot_tk)
            label.image = plot_tk  # Keep a reference
            label.pack()
        except Exception as e:
            logger.error(f"Error displaying plot: {e}")
            raise e

# =======================
# Main Execution
# =======================

def main():
    """Main function to run the EEG Image Decoder GUI"""
    # Parse command line arguments for config file
    import argparse
    parser = argparse.ArgumentParser(description="EEG Image Decoder Project")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Initialize configuration and paths
    config = Config(config_path=args.config)
    paths = ProjectPaths()
    
    # Create and run GUI
    gui = EEGImageDecoderGUI(config, paths)
    gui.root.mainloop()

if __name__ == '__main__':
    main()
