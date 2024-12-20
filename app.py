# Dream2Imagev7.py

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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import matplotlib
# Use 'Agg' backend for thread safety
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import imageio
import queue
import threading
import yaml
import json
from datetime import datetime
import pytest
from sklearn.metrics import silhouette_score
from scipy.stats import skew
from torchvision.models import resnet18
from sklearn.manifold import TSNE
import seaborn as sns
import tempfile
from torch.utils.tensorboard import SummaryWriter  # Required for TrainerVAE
import torchvision  # Required for TrainerVAE's save_reconstructions

# Suppress TensorFlow logs if not needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =======================
# Logging Setup
# =======================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('brain_decoder.log')
    ]
)
logger = logging.getLogger(__name__)

# =======================
# Configuration Handling
# =======================

class Config:
    """Handles configuration loading and defaults."""
    def __init__(self, config_path=None):
        self.config = self.default_config()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._update_config(user_config)
        elif config_path:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
        
        # Add device attribute
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def default_config(self):
        """Returns default configuration."""
        return {
            'model': {
                'latent_dim': 256,  # Increased from 128 for better representation
                'window_size': 64,
                'stride': 32,
                'batch_size': 16,   # Reduced batch size for better stability
                'learning_rate': 1e-4,  # Reduced learning rate
                'epochs': 15,
                'hidden_dims': [64, 128, 256, 512],  # Increased network capacity
                'optimizer_choice': 'AdamW',  # Changed to AdamW
                'scheduler_step_size': 20,
                'scheduler_gamma': 0.5,
                'use_dropout': False,
                'dropout_rate': 0.3,
                'use_augmentation': True,
                'augmentation_transforms': [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomRotation(15)
                ],
                'image_size': 32,
                'save_model': True,  # Correctly placed inside 'model'
                'model_save_path': 'vae_best.pth',
                'beta': 0.5,  # Introduced beta for KL divergence scaling
                'temperature': 0.07  # Added temperature for contrastive loss
            },
            'processing': {
                'eeg_channels': 'all',  # 'all' or list of channel indices
                'min_freq': 1.0,
                'max_freq': 45.0,
                'normalize': True
            },
            'training': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'random_seed': 42
            },
            'visualization': {
                'fps': 24,
                'image_size': [32, 32],  # Adjusted to 32x32
                'grid_size': [4, 4]
            },
            'pink_noise': {
                'alpha': 1.2  # Adjusted as per sleep pink noise
            }
        }
    
    def _update_config(self, user_config):
        """Recursively updates the default config with user config."""
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
    
    def __getitem__(self, item):
        return self.config.get(item, None)
    
    def __getattr__(self, item):
        return self.config.get(item, None)

# =======================
# Project Paths
# =======================

class ProjectPaths:
    """Manages project directories and paths"""
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent  # Current script directory
        self.base_dir = Path(base_dir)
        
        # Create essential directories
        self.models_dir = self.base_dir / 'models'
        self.data_dir = self.base_dir / 'data'
        self.results_dir = self.base_dir / 'results'
        self.latents_dir = self.base_dir / 'latents'
        self.eeg_dir = self.data_dir / 'eeg'
        self.images_dir = self.data_dir / 'images'
        self.processed_dir = self.data_dir / 'processed'  # Added this line
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in [
            self.base_dir, 
            self.models_dir, 
            self.data_dir,
            self.results_dir, 
            self.latents_dir, 
            self.eeg_dir, 
            self.images_dir,
            self.processed_dir  # Added this line
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name):
        return self.models_dir / f"{model_name}.pth"
    
    def get_latent_path(self, name):
        name = name.replace('.npy', '')
        return self.latents_dir / f"{name}.npy"
    
    def get_eeg_files(self):
        """Returns a list of EEG files in the eeg_dir"""
        return list(self.eeg_dir.glob('*.edf'))
    
    def get_image_files(self, subject_id):
        """Returns a list of image files for a given subject"""
        subject_image_dir = self.images_dir / f"subject{subject_id}"
        return list(subject_image_dir.glob('*.png')) + list(subject_image_dir.glob('*.jpg')) + list(subject_image_dir.glob('*.jpeg'))
# =======================
# EEG Extraction and Pairing
# =======================

class EEGExtractor:
    """Extracts and organizes sleep EEG data from EDF files with channel selection"""
    def __init__(self, edf_path, selected_channels='all'):
        self.edf_path = edf_path
        self.raw = None
        self.sleep_stages = None
        self.selected_channels = selected_channels
    
    def load_edf(self):
        """Load EDF and select channels"""
        self.raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
        
        # Handle channel selection
        if self.selected_channels != 'all':
            if isinstance(self.selected_channels, str):
                try:
                    self.selected_channels = [int(ch.strip()) for ch in self.selected_channels.split(',')]
                except ValueError:
                    raise ValueError("Channel indices must be integers separated by commas.")
            
            # Validate channel indices
            if max(self.selected_channels) >= len(self.raw.ch_names):
                raise ValueError(f"Invalid channel index: {max(self.selected_channels)} exceeds available channels.")
            
            # Select channels
            selected_ch_names = [self.raw.ch_names[i] for i in self.selected_channels]
            self.raw.pick_channels(selected_ch_names)
            logger.info(f"Selected channels: {selected_ch_names}")
        else:
            logger.info("Using all channels")
        
        # Detect sleep stages using selected channels
        data = self.raw.get_data()
        self.sleep_stages = self._detect_sleep_stages(data)
    
    def _detect_sleep_stages(self, data):
        """Enhanced sleep stage detection using selected channels"""
        stages = []
        window_size = int(self.raw.info['sfreq'] * 30)  # 30-second windows
        
        for start in range(0, data.shape[1], window_size):
            window = data[:, start:start + window_size]
            if window.shape[1] < window_size:
                break  # Discard incomplete window
                
            # Compute features for each channel
            channel_features = []
            for ch_idx in range(window.shape[0]):
                freqs, psd = welch(window[ch_idx], fs=self.raw.info['sfreq'])
                
                # Frequency bands
                delta = np.mean(psd[(freqs >= 0.5) & (freqs <= 4)])
                theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
                alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
                beta = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
                
                channel_features.append({
                    'delta': delta,
                    'theta': theta,
                    'alpha': alpha,
                    'beta': beta,
                    'delta_theta_ratio': delta/theta if theta > 0 else 0,
                    'theta_alpha_ratio': theta/alpha if alpha > 0 else 0
                })
            
            # Combine channel features for stage classification
            avg_delta_theta = np.mean([f['delta_theta_ratio'] for f in channel_features])
            avg_theta_alpha = np.mean([f['theta_alpha_ratio'] for f in channel_features])
            
            # Classify stage
            if avg_delta_theta > 2:
                stages.append('deep_sleep')
            elif avg_theta_alpha > 1.5:
                stages.append('light_sleep')
            else:
                stages.append('rem')
                
        logger.info(f"Detected sleep stages: {set(stages)} with counts { {stage: stages.count(stage) for stage in set(stages)} }")
        return stages
    
    def extract_segments(self, save_dir):
        """Extract and save EEG segments with metadata"""
        save_dir = Path(save_dir)
        metadata = {
            'channels': self.raw.ch_names,
            'sfreq': self.raw.info['sfreq'],
            'n_segments': len(self.sleep_stages),
            'stages': {},
            'segments': {}
        }
        
        # Create stage directories and count segments
        for stage in set(self.sleep_stages):
            stage_dir = save_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            metadata['stages'][stage] = 0
        
        # Extract and save segments
        window_size = int(self.raw.info['sfreq'] * 30)
        data = self.raw.get_data()
        
        for i, stage in enumerate(self.sleep_stages):
            start = i * window_size
            segment = data[:, start:start + window_size]
            
            if segment.shape[1] == window_size:  # Ensure complete segment
                segment_path = save_dir / stage / f'segment_{i}.npy'
                np.save(segment_path, segment)
                
                metadata['segments'][f'segment_{i}'] = {
                    'stage': stage,
                    'channels': self.raw.ch_names,
                    'start_time': start / self.raw.info['sfreq'],
                    'duration': 30.0  # seconds
                }
                metadata['stages'][stage] += 1
        
        # Save metadata
        with open(save_dir / 'eeg_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Extracted {len(metadata['segments'])} segments")
        logger.info(f"Stage distribution: {metadata['stages']}")
        
        # Organize CIFAR data
        self._organize_cifar_data(save_dir)
    
    def _organize_cifar_data(self, save_dir):
        """Organize CIFAR images into class folders"""
        cifar_dir = save_dir / 'cifar'
        cifar_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CIFAR-10 dataset
        trainset = datasets.CIFAR10(root='./data', train=True, download=True)
        classes = trainset.classes
        
        # Create class directories
        for cls in classes:
            (cifar_dir / cls).mkdir(parents=True, exist_ok=True)
        
        # Save images and create mapping
        mapping = {}
        for i, (img, label) in enumerate(trainset):
            cls_name = classes[label]
            img_path = cifar_dir / cls_name / f'img_{i}.png'
            img.save(img_path)
            mapping[str(i)] = {  # Changed key to string for consistency
                'class': cls_name,
                'path': str(img_path)
            }
            
            if i >= 49999:  # Limit to 50,000 images
                break
        
        # Save mapping
        with open(save_dir / 'cifar_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=2)
            
        logger.info(f"Organized CIFAR-10 images into '{cifar_dir}' with mapping saved.")
    
    def create_latent_pairs(self, save_dir, vae_model):
        """Create paired latent representations for EEG and images"""
        save_dir = Path(save_dir)
        pairs_dir = save_dir / 'latent_pairs'
        pairs_dir.mkdir(exist_ok=True)
        
        # Load metadata
        with open(save_dir / 'eeg_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load CIFAR mapping
        mapping_path = save_dir / 'cifar_mapping.json'
        if not mapping_path.exists():
            logger.error(f"CIFAR mapping file not found at '{mapping_path}'. Please ensure it exists.")
            return
        with open(mapping_path, 'r') as f:
            cifar_mapping = json.load(f)
        
        pairs = []
        for seg_id, seg_info in metadata['segments'].items():
            # Load EEG segment
            segment = np.load(save_dir / seg_info['stage'] / f"{seg_id}.npy")
            
            # Get corresponding CIFAR image (cyclic pairing)
            img_id = int(seg_id.split('_')[1]) % len(cifar_mapping)
            img_info = cifar_mapping[str(img_id)]
            
            pairs.append({
                'eeg_segment': str(seg_id),  # Ensure it's a string
                'eeg_stage': seg_info['stage'],
                'cifar_image': img_info['path'],
                'cifar_class': img_info['class']
            })
        
        # Save pairing information
        with open(pairs_dir / 'pairing_info.json', 'w') as f:
            json.dump(pairs, f, indent=2)
            
        logger.info(f"Created {len(pairs)} EEG-image pairs")

# =======================
# EEG Processing
# =======================

class EEGProcessor:
    """Handles EEG data loading and preprocessing"""
    def __init__(self, edf_path, window_size=64, stride=32, config=None):
        self.edf_path = edf_path
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = None
        self.data = None
        self.channel_names = None
        self.config = config
    
    def load_and_preprocess(self):
        """Load and preprocess EEG data"""
        logger.info(f"Loading EEG data from {self.edf_path}")
        
        # Load raw EEG
        raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
        self.sampling_rate = raw.info['sfreq']
        self.channel_names = raw.ch_names
        
        # Select channels
        eeg_channels = self.config['processing']['eeg_channels']
        if eeg_channels != 'all':
            if isinstance(eeg_channels, list):
                # Validate channel indices
                if max(eeg_channels) >= len(self.channel_names):
                    raise ValueError(f"Invalid channel index: {max(eeg_channels)} exceeds available channels.")
                selected_channels = [self.channel_names[i] for i in eeg_channels]
                raw.pick_channels(selected_channels)
                logger.info(f"Selected channels: {selected_channels}")
            else:
                raise ValueError("eeg_channels should be 'all' or a list of channel indices")
        else:
            logger.info("Using all channels.")
        
        # Filter data
        nyquist = self.sampling_rate / 2.0
        high_freq = min(self.config['processing']['max_freq'], nyquist - 5.0)  # Stay below Nyquist
        raw.filter(self.config['processing']['min_freq'], high_freq, fir_design='firwin')
        logger.info(f"Filtered EEG data between {self.config['processing']['min_freq']}Hz and {high_freq}Hz")
        
        # Get filtered data
        self.data = raw.get_data()
        
        # Normalize
        if self.config['processing']['normalize']:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data.T).T
            logger.info("Normalized EEG data.")
        
        logger.info(f"Preprocessed EEG data shape: {self.data.shape}")
        return self.data
    
    def extract_windows(self):
        """Extract fixed-size windows from EEG data"""
        if self.data is None:
            raise ValueError("Call load_and_preprocess first")
            
        windows = []
        n_channels, n_samples = self.data.shape
        
        for start in range(0, n_samples - self.window_size, self.stride):
            window = self.data[:, start:start + self.window_size]
            windows.append(window)
            
        logger.info(f"Extracted {len(windows)} windows from EEG data.")
        return np.array(windows)

# =======================
# Pink Noise Utility
# =======================

def generate_pink_noise(n_samples, alpha=1.0):
    """
    Generates pink noise (1/f^alpha) for a given number of samples.
    Args:
        n_samples (int): Number of samples to generate.
        alpha (float): Exponent for the power law.
    Returns:
        np.ndarray: Pink noise array.
    """
    # Generate white noise
    white = np.random.randn(n_samples)
    
    # Perform FFT
    fft = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0)
    
    # Avoid division by zero
    frequencies[0] = frequencies[1]
    
    # Apply 1/f^alpha
    fft *= 1 / (frequencies ** (alpha / 2.0))
    
    # Inverse FFT
    pink = np.fft.irfft(fft, n=n_samples)
    
    # Normalize
    pink = pink / np.std(pink)
    return pink

# =======================
# Sleep Pink Noise Loss
# =======================

class SleepPinkNoiseLoss(nn.Module):
    """Simplified pink noise analysis for sleep EEG from posterior electrode"""
    def __init__(self, alpha_sleep=1.2):
        super().__init__()
        self.alpha_sleep = alpha_sleep
        
    def detect_sleep_depth(self, signal):
        """Estimate sleep depth from signal characteristics"""
        # Detach tensor before converting to numpy
        signal_np = signal.detach().cpu().numpy()
        
        # Compute relative power in different frequency bands
        freq, psd = welch(signal_np, fs=256, nperseg=256)
        
        # Define frequency bands
        delta_mask = (freq >= 0.5) & (freq <= 4)
        theta_mask = (freq >= 4) & (freq <= 8)
        alpha_mask = (freq >= 8) & (freq <= 13)
        
        # Calculate band powers
        delta_power = np.mean(psd[:, delta_mask], axis=1)
        theta_power = np.mean(psd[:, theta_mask], axis=1)
        alpha_power = np.mean(psd[:, alpha_mask], axis=1)
        
        # High delta/alpha ratio suggests deeper sleep
        sleep_depth = delta_power / (alpha_power + 1e-6)
        return torch.tensor(sleep_depth, device=signal.device)
    
    def forward(self, latent, image_latent):
        """
        Compute pink noise deviation with sleep-state-dependent scaling
        """
        # Detect sleep depth
        sleep_depth = self.detect_sleep_depth(latent)
        
        # Adjust alpha based on sleep depth
        adjusted_alpha = self.alpha_sleep * (1 + 0.2 * torch.sigmoid(sleep_depth))
        
        # Detach tensors before converting to numpy
        latent_np = latent.detach().cpu().numpy()
        image_latent_np = image_latent.detach().cpu().numpy()
        
        # Compute power spectral density
        psd_latent = []
        psd_image = []
        for i in range(latent_np.shape[1]):
            freq, power = welch(latent_np[:, i], fs=256, nperseg=min(256, latent_np.shape[1]))
            psd_latent.append(power)
            freq_img, power_img = welch(image_latent_np[:, i], fs=256, nperseg=min(256, image_latent_np.shape[1]))
            psd_image.append(power_img)
            
        psd_latent = np.array(psd_latent)
        psd_image = np.array(psd_image)
        mean_psd_latent = np.mean(psd_latent[:, 1:], axis=0)
        mean_psd_image = np.mean(psd_image[:, 1:], axis=0)
        
        # Expected PSD for adjusted pink noise
        freq = freq[1:]
        expected_psd = 1.0 / (freq ** adjusted_alpha.mean().item())
        expected_psd = expected_psd / np.mean(expected_psd)
        
        # Compute actual PSD
        psd_mean = mean_psd_latent / np.mean(mean_psd_latent)
        psd_image_mean = mean_psd_image / np.mean(mean_psd_image)
        
        # Compare in log space
        loss_latent = torch.tensor(np.mean((np.log(psd_mean + 1e-8) - 
                                   np.log(expected_psd + 1e-8)) ** 2),
                          requires_grad=True).to(latent.device)
        loss_image = torch.tensor(np.mean((np.log(psd_image_mean + 1e-8) - 
                                   np.log(expected_psd + 1e-8)) ** 2),
                          requires_grad=True).to(latent.device)
        
        return loss_latent + loss_image, {
            'sleep_depth': sleep_depth,
            'adjusted_alpha': adjusted_alpha,
            'psd_ratio_latent': psd_mean / expected_psd,
            'psd_ratio_image': psd_image_mean / expected_psd
        }

# =======================
# Sleep Image Encoder and Loss
# =======================

class SleepImageEncoder(nn.Module):
    """
    Encodes sleep EEG signals into an image-compatible latent space.
    Enhanced with Transformer layers and Spatial Attention for better feature extraction.
    """
    def __init__(self, num_channels, seq_length, latent_dim=256):
        super().__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # Temporal feature extraction with CNNs
        self.freq_conv = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(256),
            nn.ELU()
        )
        
        # Spatial Attention for multi-channel EEG
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.Sigmoid()
        )
        
        # Transformer for temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Sleep stage-aware processing with Transformer
        stage_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
            dropout=0.1
        )
        self.stage_encoder = nn.TransformerEncoder(stage_layer, num_layers=1)
        
        # Project to latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim * 2)
        )
        
        # Frequency band embedding
        self.freq_embedding = nn.Sequential(
            nn.Linear(5, 64),
            nn.ELU(),
            nn.Linear(64, 256)
        )
        
    def reshape_input(self, x):
        """Reshape input to [batch, channels, sequence]"""
        if len(x.shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = x.shape
            return x.view(batch_size, channels, height * width)
        elif len(x.shape) == 3:  # [batch, channels, sequence]
            return x
        elif len(x.shape) == 2:  # [batch, sequence]
            return x.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
    
    def extract_frequency_features(self, x):
        """Extract power in different frequency bands using FFT"""
        # Compute FFT
        fft = torch.fft.rfft(x, dim=-1)
        power = torch.abs(fft) ** 2
        
        # Define frequency bands (assuming 256Hz sampling rate)
        delta = torch.mean(power[..., 1:4], dim=-1)
        theta = torch.mean(power[..., 4:8], dim=-1)
        alpha = torch.mean(power[..., 8:13], dim=-1)
        beta = torch.mean(power[..., 13:30], dim=-1)
        gamma = torch.mean(power[..., 30:45], dim=-1)
        
        bands = torch.stack([delta, theta, alpha, beta, gamma], dim=-1)
        return self.freq_embedding(bands)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Reshape input
        x = self.reshape_input(x)
        batch_size = x.shape[0]
        
        # Extract temporal features
        temporal_features = self.freq_conv(x)  # [batch, 256, sequence]
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(temporal_features)
        temporal_features = temporal_features * attention_weights
        
        # Prepare for transformer (batch_first=True)
        features = temporal_features.transpose(1, 2)  # [batch, sequence, 256]
        
        # Apply transformer
        transformer_out = self.transformer(features)  # [batch, sequence, 256]
        
        # Apply sleep stage-aware transformer
        stage_out = self.stage_encoder(transformer_out)  # [batch, sequence, 256]
        
        # Global average pooling
        pooled = torch.mean(stage_out, dim=1)  # [batch, 256]
        
        # Project to latent space
        latent_params = self.latent_projection(pooled)
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

class Reshape(nn.Module):
    """Custom reshape layer to ensure consistent dimensions"""
    def forward(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 4:  # [batch, channels, height, width]
            # Flatten height and width into sequence length
            return x.view(batch_size, x.size(1), -1)
        elif len(x.shape) == 3:  # [batch, channels, sequence]
            return x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

class SleepEncoderLoss(nn.Module):
    """Custom loss function for training the sleep encoder"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, mu, logvar, sleep_consistency=None):
        """
        Compute the loss with optional sleep stage consistency
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            sleep_consistency: Optional tensor of shape (batch_size,) indicating
                             confidence in sleep stage classification
        """
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # If sleep stage confidence is provided, weight the KL loss
        if sleep_consistency is not None:
            kl_loss = kl_loss * sleep_consistency.mean()
        
        # Scale KL loss
        kl_loss = self.beta * kl_loss
        
        return kl_loss

# =======================
# VAE Model Definition with Skip Connections
# =======================

class VAE(nn.Module):
    """Variational Autoencoder for image reconstruction with skip connections"""
    def __init__(self, config: Config):
        super(VAE, self).__init__()
        self.latent_dim = config.config['model']['latent_dim']
        self.hidden_dims = config.config['model']['hidden_dims']
        self.use_dropout = config.config['model']['use_dropout']
        self.dropout_rate = config.config['model']['dropout_rate']

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        in_channels = 3
        for h_dim in self.hidden_dims:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU()
            )
            if self.use_dropout:
                conv.add_module("dropout", nn.Dropout(self.dropout_rate))
            self.encoder_layers.append(conv)
            in_channels = h_dim

        self.flatten_dim = self.hidden_dims[-1] * (32 // 2**len(self.hidden_dims))**2
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)

        # Decoder layers
        self.decoder_input = nn.Linear(self.latent_dim, self.flatten_dim)
        self.decoder_layers = nn.ModuleList()
        hidden_dims_rev = list(reversed(self.hidden_dims))  # e.g., [512, 256, 128, 64]
        for i in range(len(hidden_dims_rev) - 1):
            deconv = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_rev[i], hidden_dims_rev[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims_rev[i+1]),
                nn.ReLU()
            )
            if self.use_dropout and i != len(hidden_dims_rev) - 2:
                deconv.add_module("dropout", nn.Dropout(self.dropout_rate))
            self.decoder_layers.append(deconv)
        
        # Final layer to upscale from 8x8 to 16x16
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims_rev[-1], 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Ensures output is in [0,1]
        )
        
        # Skip connection projections
        self.skip_projections = nn.ModuleList()
        for i in range(len(hidden_dims_rev) -1):
            # Projection from encoder hidden_dims to decoder hidden_dims_rev[i+1]
            skip_dim = self.hidden_dims[-(i+1)]
            decoder_dim = hidden_dims_rev[i+1]
            self.skip_projections.append(nn.Conv2d(skip_dim, decoder_dim, kernel_size=1))

    def encode(self, x):
        skip_connections = []
        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, skip_connections

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE-style encoding"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skip_connections):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[-1], 2, 2)  # Adjust based on hidden_dims and image size
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # Add skip connection with projection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                if skip.shape != x.shape:
                    # Adjust spatial dimensions
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
                # Project skip to match decoder channels
                skip = self.skip_projections[i](skip)
                x = x + skip
        x = self.final_layer(x)
        return x

    def forward(self, x):
        mu, logvar, skip_connections = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skip_connections), mu, logvar

# =======================
# Image Encoder
# =======================

class ImageEncoder(nn.Module):
    """Encoder for extracting image features with Transformer integration"""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # [32,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # [32,16,16]
            nn.Conv2d(32, 64, 3, padding=1), # [64,16,16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # [64,8,8]
            nn.Conv2d(64, 128, 3, padding=1),# [128,8,8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # [128,4,4]
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(),
            # Transformer Encoder
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8), num_layers=2)
        )
        
    def forward(self, x):
        return self.encoder(x)

# =======================
# Modality Alignment Encoder
# =======================

class ModalityAlignmentEncoder(nn.Module):
    """
    Encoder to align EEG and Image latent spaces using adversarial training.
    Uses a domain discriminator to enforce similar distributions.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, latent):
        return self.domain_discriminator(latent)

# =======================
# Brain Decoder Model
# =======================

class BrainDecoderModel(nn.Module):
    """Complete model for brain-to-image decoding with enhanced encoders and modality alignment"""
    def __init__(self, num_channels, seq_length, config: Config):
        super().__init__()
        
        # EEG Encoder
        self.eeg_encoder = SleepImageEncoder(
            num_channels=num_channels,
            seq_length=seq_length,
            latent_dim=config.config['model']['latent_dim']
        )
        
        # Image Encoder
        self.image_encoder = ImageEncoder(
            latent_dim=config.config['model']['latent_dim']
        )
        
        # Modality Alignment Encoder
        self.modality_alignment = ModalityAlignmentEncoder(
            latent_dim=config.config['model']['latent_dim']
        )
        
        # Image Decoder
        self.image_decoder = nn.Sequential(
            nn.Linear(config.config['model']['latent_dim'], 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3 * 32 * 32),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, 32, 32))
        )
    
    def forward(self, eeg, images):
        # Ensure EEG input is the right shape
        if len(eeg.shape) == 3:
            eeg = eeg.unsqueeze(1)  # Add extra dimension if needed
        
        # Process through EEG encoder
        eeg_features, eeg_mu, eeg_logvar = self.eeg_encoder(eeg)
        
        # Process through Image encoder
        image_features = self.image_encoder(images)
        
        # Generate images
        generated_images = self.image_decoder(eeg_features)
        
        return generated_images, eeg_features, image_features
    
    def generate(self, eeg):
        """Generate images from EEG input"""
        with torch.no_grad():
            eeg_features, _, _ = self.eeg_encoder(eeg)
            generated_images = self.image_decoder(eeg_features)
        return generated_images

# =======================
# Trainer Classes
# =======================

class TrainerVAE:
    """Handles VAE model training and evaluation"""
    def __init__(self, model, device, project_paths, train_loader, val_loader, config: Config):
        self.model = model
        self.device = device
        self.paths = project_paths
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.early_stopping_patience = 10
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.writer = SummaryWriter(log_dir=self.paths.results_dir / 'tensorboard')  # Requires TensorBoard installed
    
    def _get_optimizer(self):
        optimizer_choice = self.config.config['model']['optimizer_choice']
        lr = self.config.config['model']['learning_rate']
        if optimizer_choice == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_choice == 'AdamW':
            return torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_choice == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer choice")
    
    def _get_scheduler(self):
        step_size = self.config.config['model']['scheduler_step_size']
        gamma = self.config.config['model']['scheduler_gamma']
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta = self.config.config['model']['beta']  # Introduced beta for KL divergence scaling
        return BCE + beta * KLD
    
    def save_reconstructions(self, epoch, data_loader):
        """Save sample reconstructions from the VAE"""
        self.model.eval()
        with torch.no_grad():
            # Get a batch of training data
            for data, _ in data_loader:
                data = data.to(self.device)
                recon, _, _ = self.model(data)
                break  # Take only the first batch
            
            # Move to CPU and convert to numpy
            data = data.cpu()
            recon = recon.cpu()
            
            # Create a grid of original and reconstructed images
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon[:n]])
            grid = torchvision.utils.make_grid(comparison, nrow=n)
            
            # Convert to numpy
            np_grid = grid.numpy().transpose((1, 2, 0))
            
            # Plot and save
            plt.figure(figsize=(20, 4))
            plt.imshow(np_grid)
            plt.title(f'Epoch {epoch} Original (Left) vs Reconstructed (Right)')
            plt.axis('off')
            save_path = self.paths.results_dir / f'reconstructions_epoch_{epoch}.png'
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved reconstructions to '{save_path}'")
        
        self.model.train()
    
    def train(self):
        self.model.to(self.device)
        self.model.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(1, self.config.config['model']['epochs'] + 1):
            logger.info(f"Epoch {epoch}/{self.config.config['model']['epochs']}")
            train_loss = 0
            for batch_idx, (data, _) in enumerate(tqdm(self.train_loader, desc=f"VAE Training Epoch {epoch}")):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                
                if batch_idx % 100 == 0:
                    logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                                f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
            
            avg_train_loss = train_loss / len(self.train_loader.dataset)
            train_losses.append(avg_train_loss)
            logger.info(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            
            # Save sample reconstructions
            if epoch % 10 == 0 or epoch == 1:
                self.save_reconstructions(epoch, self.train_loader)
            
            # Validation
            val_loss = self.validate()
            avg_val_loss = val_loss / len(self.val_loader.dataset)
            val_losses.append(avg_val_loss)
            logger.info(f'====> Epoch: {epoch} Average validation loss: {avg_val_loss:.4f}')
            self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            
            # Checkpointing
            if avg_val_loss < self.best_val_loss and self.config.config['model']['save_model']:
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.paths.get_model_path('vae_best'))
                logger.info(f"Saved best VAE model with validation loss: {self.best_val_loss:.4f}")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), self.paths.get_model_path(f'vae_epoch_{epoch}'))
                logger.info(f"Saved VAE model checkpoint at epoch {epoch}.")
            
            # Scheduler step
            self.scheduler.step()
        
        # Plot loss curves
        plt.figure(figsize=(10,5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Training and Validation Loss')
        plt.legend()
        plt.savefig(self.paths.results_dir / 'vae_loss_curves.png')
        plt.close()
        
        self.writer.close()
        logger.info("VAE training completed.")
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in tqdm(self.val_loader, desc="VAE Validation", leave=False):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        self.model.train()
        return val_loss

class Trainer:
    """Handles model training and evaluation for BrainDecoderModel"""
    def __init__(self, model, device, project_paths, image_encoder, contrastive_loss, config: Config):
        self.model = model
        self.device = device
        self.paths = project_paths
        self.image_encoder = image_encoder
        self.contrastive_loss = contrastive_loss
        self.config = config
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.best_val_loss = float('inf')
        
        # Initialize Image Encoder Optimizer
        self.image_encoder_optimizer = torch.optim.AdamW(
            self.image_encoder.parameters(),
            lr=self.config.config['model']['learning_rate']
        )
        
        # Initialize Modality Alignment Encoder Optimizer
        self.modality_alignment_optimizer = torch.optim.AdamW(
            self.model.modality_alignment.parameters(),
            lr=self.config.config['model']['learning_rate']
        )
    
    def _get_optimizer(self):
        optimizer_choice = self.config.config['model']['optimizer_choice']
        lr = self.config.config['model']['learning_rate']
        if optimizer_choice == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_choice == 'AdamW':
            return torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_choice == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer choice")
    
    def _get_scheduler(self):
        step_size = self.config.config['model']['scheduler_step_size']
        gamma = self.config.config['model']['scheduler_gamma']
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    
    def train_epoch(self, train_loader):
        self.model.train()
        self.image_encoder.train()
        self.model.modality_alignment.train()
        total_loss = 0
        contrast_loss_total = 0
        pink_loss_total = 0
        encoder_align_total = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for eeg_batch, image_batch, _ in pbar:
                eeg_batch = eeg_batch.to(self.device)
                image_batch = image_batch.to(self.device)
                
                # Zero gradients for all optimizers
                self.optimizer.zero_grad()
                self.image_encoder_optimizer.zero_grad()
                self.modality_alignment_optimizer.zero_grad()
                
                # Forward pass
                generated_images, eeg_features, image_features = self.model(eeg_batch, image_batch)
                
                # Get image encoder features
                encoded_images = self.image_encoder(image_batch)
                
                # Compute contrastive loss
                loss, stats = self.contrastive_loss(eeg_features, encoded_images)
                
                # Additional loss for image encoder alignment
                encoder_alignment_loss = F.mse_loss(encoded_images, image_features)
                total_loss = loss + encoder_alignment_loss
                
                # Adversarial loss for modality alignment
                domain_labels_eeg = torch.ones(eeg_features.size(0), 1).to(self.device)
                domain_labels_image = torch.zeros(image_features.size(0), 1).to(self.device)
                
                # Predict domains
                domain_pred_eeg = self.model.modality_alignment(eeg_features)
                domain_pred_image = self.model.modality_alignment(encoded_images)
                
                # Compute adversarial loss
                adversarial_loss_eeg = F.binary_cross_entropy(domain_pred_eeg, domain_labels_eeg)
                adversarial_loss_image = F.binary_cross_entropy(domain_pred_image, domain_labels_image)
                adversarial_loss = adversarial_loss_eeg + adversarial_loss_image
                
                # Total loss
                final_loss = total_loss + adversarial_loss
                
                # Backward pass
                final_loss.backward()
                self.optimizer.step()
                self.image_encoder_optimizer.step()
                self.modality_alignment_optimizer.step()
                
                # Update statistics
                total_loss_value = final_loss.item()
                contrast_loss_total += stats['sleep_depth'].sum().item()
                pink_loss_total += stats['adjusted_alpha'].mean().item()
                encoder_align_total += encoder_alignment_loss.item()
                
                pbar.set_postfix({
                    'Total Loss': total_loss_value, 
                    'Contrast': stats['sleep_depth'].mean().item(), 
                    'Pink Noise': stats['adjusted_alpha'].mean().item(),
                    'Encoder Align': encoder_alignment_loss.item(),
                    'Adversarial': adversarial_loss.item()
                })
        
        avg_loss = total_loss_value / len(train_loader)
        avg_contrast = contrast_loss_total / len(train_loader)
        avg_pink = pink_loss_total / len(train_loader)
        avg_encoder_align = encoder_align_total / len(train_loader)
        return avg_loss, avg_contrast, avg_pink, avg_encoder_align
    
    def evaluate(self, val_loader):
        self.model.eval()
        self.image_encoder.eval()
        self.model.modality_alignment.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for eeg_batch, image_batch, _ in tqdm(val_loader, desc='Validation', leave=False):
                eeg_batch = eeg_batch.to(self.device)
                image_batch = image_batch.to(self.device)
                
                # Forward pass
                generated_images, eeg_features, image_features = self.model(eeg_batch, image_batch)
                encoded_images = self.image_encoder(image_batch)
                
                # Compute contrastive loss
                loss, _ = self.contrastive_loss(eeg_features, encoded_images)
                encoder_alignment_loss = F.mse_loss(encoded_images, image_features)
                
                # Adversarial loss
                domain_labels_eeg = torch.ones(eeg_features.size(0), 1).to(self.device)
                domain_labels_image = torch.zeros(image_features.size(0), 1).to(self.device)
                domain_pred_eeg = self.model.modality_alignment(eeg_features)
                domain_pred_image = self.model.modality_alignment(encoded_images)
                adversarial_loss_eeg = F.binary_cross_entropy(domain_pred_eeg, domain_labels_eeg)
                adversarial_loss_image = F.binary_cross_entropy(domain_pred_image, domain_labels_image)
                adversarial_loss = adversarial_loss_eeg + adversarial_loss_image
                
                # Total loss
                total_loss = loss + encoder_alignment_loss + adversarial_loss
                
                val_loss += total_loss.item()
                
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        train_contrasts = []
        train_pinks = []
        train_encoder_aligns = []
        
        for epoch in range(1, self.config.config['model']['epochs'] + 1):
            logger.info(f"Epoch {epoch}/{self.config.config['model']['epochs']}")
            
            # Training
            train_loss, train_contrast, train_pink, train_encoder_align = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_contrasts.append(train_contrast)
            train_pinks.append(train_pink)
            train_encoder_aligns.append(train_encoder_align)
            
            # Validation
            val_loss = self.evaluate(val_loader)
            val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Contrast: {train_contrast:.4f} | Pink Noise: {train_pink:.4f} | Encoder Align: {train_encoder_align:.4f}")
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Save checkpoints if improved
            if val_loss < self.best_val_loss and self.config.config['model']['save_model']:
                self.best_val_loss = val_loss
                
                # Save brain decoder model
                torch.save(self.model.state_dict(), 
                         self.paths.get_model_path('brain_decoder_best'))
                
                # Save image encoder model
                torch.save(self.image_encoder.state_dict(), 
                         self.paths.get_model_path('image_encoder_best'))
                
                # Save modality alignment encoder model
                torch.save(self.model.modality_alignment.state_dict(),
                           self.paths.get_model_path('modality_alignment_best'))
                
                logger.info(f"Saved best models with validation loss: {self.best_val_loss:.4f}")
            
            # Scheduler step
            self.scheduler.step()
        
        # Plot loss curves
        plt.figure(figsize=(10,5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Brain Decoder Training and Validation Loss')
        plt.legend()
        plt.savefig(self.paths.results_dir / 'brain_decoder_loss_curves.png')
        plt.close()
        
        logger.info("Training completed.")

# =======================
# Result Analyzer
# =======================

class ResultAnalyzer:
    """Analyzes and visualizes brain decoding results"""
    def __init__(self, project_paths, device):
        self.paths = project_paths
        self.device = device
        self.feature_extractor = self._load_feature_extractor()
    
    def _load_feature_extractor(self):
        """Load pretrained ResNet for feature extraction"""
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device)
        model.eval()
        return model
    
    def compute_metrics(self, generated_images, target_images):
        """Compute various quality metrics"""
        # Convert lists to numpy arrays if needed
        if isinstance(generated_images, list):
            generated_images = np.array(generated_images)
        if isinstance(target_images, list):
            target_images = np.array(target_images)
            
        metrics = {
            'pixel_mse': self._compute_pixel_mse(generated_images, target_images),
            'ssim': self._compute_ssim(generated_images, target_images),
            'perceptual_distance': self._compute_perceptual_distance(generated_images, target_images),
            'feature_correlation': self._compute_feature_correlation(generated_images, target_images)
        }
        return metrics
    
    def _compute_perceptual_distance(self, generated, target):
        """Compute perceptual distance using feature extractor"""
        with torch.no_grad():
            # Convert to numpy arrays first if they're lists
            generated = np.array(generated)
            target = np.array(target)
            
            # Convert to tensors and move to device
            gen_tensor = torch.tensor(generated).float().to(self.device)
            tgt_tensor = torch.tensor(target).float().to(self.device)
            
            # Reshape if needed
            if len(gen_tensor.shape) == 3:
                gen_tensor = gen_tensor.unsqueeze(0)
            if len(tgt_tensor.shape) == 3:
                tgt_tensor = tgt_tensor.unsqueeze(0)
                
            # Extract features
            gen_features = self._extract_features(gen_tensor)
            tgt_features = self._extract_features(tgt_tensor)
            
            # Compute distance
            distance = F.mse_loss(gen_features, tgt_features).item()
        
        return distance

    def _extract_features(self, images):
        """Extract features using pretrained model"""
        if images.shape[-3:] != (3, 32, 32):
            # Ensure correct shape (B, C, H, W)
            images = images.permute(0, 3, 1, 2) if len(images.shape) == 4 else images.permute(2, 0, 1).unsqueeze(0)
        
        # Ensure values are in [0, 1]
        if images.max() > 1:
            images = images / 255.0
            
        images = images.to(self.device)
        features = self.feature_extractor(images)
        return features
    
    def _compute_pixel_mse(self, generated, target):
        """Compute pixel-wise MSE"""
        # Convert lists to numpy arrays first
        generated_array = np.array(generated)
        target_array = np.array(target)
        # Then convert to tensor
        return F.mse_loss(torch.tensor(generated_array), 
                        torch.tensor(target_array)).item()
    
    def _compute_ssim(self, generated, target):
        """Compute Structural Similarity Index"""
        from skimage.metrics import structural_similarity as compare_ssim
        ssim_values = []
        for gen, tgt in zip(generated, target):
            # Convert to grayscale for SSIM
            gen_gray = cv2.cvtColor(gen.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            tgt_gray = cv2.cvtColor(tgt.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            ssim, _ = compare_ssim(gen_gray, tgt_gray, full=True)
            ssim_values.append(ssim)
        return np.mean(ssim_values)
    
    def _compute_ssim(self, generated, target):
        """Compute Structural Similarity Index"""
        from skimage.metrics import structural_similarity as compare_ssim
        ssim_values = []
        for gen, tgt in zip(generated, target):
            # Convert to grayscale for SSIM
            # Ensure proper scaling and dtype conversion
            gen_gray = cv2.cvtColor((gen.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), 
                                cv2.COLOR_RGB2GRAY)
            tgt_gray = cv2.cvtColor((tgt.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), 
                                cv2.COLOR_RGB2GRAY)
            ssim, _ = compare_ssim(gen_gray, tgt_gray, full=True, data_range=255)  # Added data_range
            ssim_values.append(ssim)
        return np.mean(ssim_values)
    
    def _compute_feature_correlation(self, generated, target):
        """Compute correlation between extracted features"""
        with torch.no_grad():
            # Convert to numpy arrays first if they're lists
            generated = np.array(generated)
            target = np.array(target)
            
            # Convert to tensors and ensure correct shape
            if len(generated.shape) == 4 and generated.shape[1] == 32:
                # If shape is [batch, 32, 3, 32], permute to [batch, 3, 32, 32]
                gen_tensor = torch.tensor(generated).float().permute(0, 2, 1, 3)
            else:
                gen_tensor = torch.tensor(generated).float()
                
            if len(target.shape) == 4 and target.shape[1] == 32:
                # If shape is [batch, 32, 3, 32], permute to [batch, 3, 32, 32]
                tgt_tensor = torch.tensor(target).float().permute(0, 2, 1, 3)
            else:
                tgt_tensor = torch.tensor(target).float()
                
            # Extract features
            gen_features = self._extract_features(gen_tensor.to(self.device))
            tgt_features = self._extract_features(tgt_tensor.to(self.device))
            
            # Flatten features and compute correlation
            gen_features = gen_features.cpu().numpy().flatten()
            tgt_features = tgt_features.cpu().numpy().flatten()
            correlation = np.corrcoef(gen_features, tgt_features)[0, 1]
            
        return correlation
    
    def _extract_features(self, images):
        """Extract features using pretrained model"""
        # Ensure correct shape (B, C, H, W)
        if images.shape[-3:] != (3, 32, 32):
            if len(images.shape) == 4:
                # Handle case where channels might be in wrong position
                if images.shape[1] == 32:
                    images = images.permute(0, 2, 1, 3)
                elif images.shape[2] == 3:
                    images = images.permute(0, 2, 3, 1)
            elif len(images.shape) == 3:
                images = images.permute(2, 0, 1).unsqueeze(0)
        
        # Ensure values are in [0, 1]
        if images.max() > 1:
            images = images / 255.0
        
        # Add batch dimension if needed
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        # Debug output
        logger.info(f"Image shape before feature extraction: {images.shape}")
        
        # Process through feature extractor
        images = images.to(self.device)
        features = self.feature_extractor(images)
        
        return features
    
    def analyze_latent_space(self, eeg_latents, image_latents, labels=None):
        """Analyze latent space structure with graceful error handling"""
        try:
            analysis = {
                'clustering': self._analyze_clustering(eeg_latents, labels),
                'distribution': self._analyze_distribution(eeg_latents),
                'alignment': self._analyze_alignment(eeg_latents, image_latents)
            }
        except Exception as e:
            logger.error(f"Error in latent space analysis: {str(e)}")
            # Return partial analysis if possible
            analysis = {
                'clustering': None,
                'distribution': self._analyze_distribution(eeg_latents),
                'alignment': self._analyze_alignment(eeg_latents, image_latents)
            }
            
        return analysis
    
    def _analyze_clustering(self, latents, labels):
        """Analyze clustering structure in latent space with graceful handling of edge cases"""
        if labels is None:
            logger.info("No labels provided for clustering analysis")
            return None
            
        # Check number of unique labels
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            logger.info(f"Only {n_clusters} cluster(s) found - skipping silhouette score")
            return {
                'silhouette_score': None,
                'k_star_distribution': self._compute_k_star_distribution(latents, labels),
                'n_clusters': n_clusters,
                'cluster_sizes': {str(label): np.sum(labels == label) for label in unique_labels}
            }
        
        try:
            silhouette = silhouette_score(latents, labels)
        except Exception as e:
            logger.warning(f"Failed to compute silhouette score: {str(e)}")
            silhouette = None
        
        # Compute k* distribution even if silhouette fails
        k_star_dist = self._compute_k_star_distribution(latents, labels)
        
        return {
            'silhouette_score': silhouette,
            'k_star_distribution': k_star_dist,
            'n_clusters': n_clusters,
            'cluster_sizes': {str(label): np.sum(labels == label) for label in unique_labels}
        }
    
    def _compute_k_star_distribution(self, latents, labels):
        """Compute k* distribution for latent space analysis"""
        k_star_values = []
        
        for i, point in enumerate(latents):
            # Compute distances to all other points
            distances = np.linalg.norm(latents - point, axis=1)
            
            # Sort indices by distance
            sorted_indices = np.argsort(distances)
            
            # Find first index with different label
            # Exclude the point itself by starting from index 1
            k_star = np.where(labels[sorted_indices[1:]] != labels[i])[0]
            if len(k_star) > 0:
                k_star = k_star[0] + 1  # Adding 1 because we skipped the first index
            else:
                k_star = len(latents) - 1  # If all labels are same
            k_star_values.append(k_star)
            
        k_star_values = np.array(k_star_values) / len(latents)
        
        return {
            'values': k_star_values,
            'mean': np.mean(k_star_values),
            'std': np.std(k_star_values),
            'skew': skew(k_star_values)
        }
    
    def _analyze_distribution(self, latents):
        """Analyze the distribution of latent vectors"""
        return {
            'mean': np.mean(latents, axis=0),
            'std': np.std(latents, axis=0),
            'skew': skew(latents, axis=0)
        }
    
    def _analyze_alignment(self, eeg_latents, image_latents):
        """Analyze alignment between EEG and image latent spaces with size handling"""
        try:
            # Log shapes for debugging
            logger.info(f"EEG latents shape: {eeg_latents.shape}")
            logger.info(f"Image latents shape: {image_latents.shape}")
            
            # Ensure we're using the same number of samples
            min_samples = min(eeg_latents.shape[0], image_latents.shape[0])
            eeg_latents = eeg_latents[:min_samples]
            image_latents = image_latents[:min_samples]
            
            # Ensure we're comparing the same dimensionality
            if eeg_latents.shape[1] != image_latents.shape[1]:
                logger.warning("Latent dimensions don't match - using dimensionality reduction")
                # Use smaller dimension as target
                target_dim = min(eeg_latents.shape[1], image_latents.shape[1])
                
                # Simple dimensionality reduction via PCA
                from sklearn.decomposition import PCA
                
                if eeg_latents.shape[1] > target_dim:
                    pca = PCA(n_components=target_dim)
                    eeg_latents = pca.fit_transform(eeg_latents)
                    
                if image_latents.shape[1] > target_dim:
                    pca = PCA(n_components=target_dim)
                    image_latents = pca.fit_transform(image_latents)
            
            # Compute correlation matrix
            correlation = np.corrcoef(eeg_latents.T, image_latents.T)
            n = len(eeg_latents.T)
            correlation = correlation[:n, n:]
            
            return {
                'mean_correlation': np.mean(correlation),
                'max_correlation': np.max(correlation),
                'correlation_matrix': correlation,
                'eeg_dim': eeg_latents.shape[1],
                'image_dim': image_latents.shape[1],
                'n_samples': min_samples
            }
            
        except Exception as e:
            logger.error(f"Error in alignment analysis: {str(e)}")
            return {
                'mean_correlation': None,
                'max_correlation': None,
                'correlation_matrix': None,
                'error': str(e)
            }
    
    def visualize_results(self, analysis, save_dir=None):
        """Create visualization of analysis results"""
        if save_dir is None:
            save_dir = self.paths.results_dir / 'analysis'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot latent space visualization
        self._plot_latent_space(analysis, save_dir)
        
        # Plot k* distribution
        self._plot_k_star_distribution(analysis, save_dir)
        
        # Plot correlation matrix
        self._plot_correlation_matrix(analysis, save_dir)
    
    def _plot_latent_space(self, analysis, save_dir):
        """Plot t-SNE visualization of latent space"""
        if 'clustering' not in analysis or analysis['clustering'] is None:
            return
            
        # Assuming 'k_star_distribution' contains 'values'
        tsne = TSNE(n_components=2, random_state=42)
        eeg_embedded = tsne.fit_transform(analysis['clustering']['k_star_distribution']['values'].reshape(-1, 1))
        
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(eeg_embedded[:, 0], eeg_embedded[:, 1],
                            c=analysis['clustering']['silhouette_score'],
                            cmap='viridis')
        plt.colorbar(scatter, label='Silhouette Score')
        plt.title('Latent Space Visualization (t-SNE)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(save_dir / 'latent_space.png')
        plt.close()
    
    def _plot_k_star_distribution(self, analysis, save_dir):
        """Plot k* distribution"""
        if 'clustering' not in analysis or 'k_star_distribution' not in analysis['clustering']:
            return
            
        k_star = analysis['clustering']['k_star_distribution']
        plt.figure(figsize=(10, 6))
        sns.histplot(k_star['values'], bins=50, kde=True)
        plt.axvline(k_star['mean'], color='r', linestyle='--',
                   label=f"Mean: {k_star['mean']:.3f}")
        plt.title(f"k* Distribution (skew: {k_star['skew']:.3f})")
        plt.xlabel('k*')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(save_dir / 'k_star_distribution.png')
        plt.close()
    
    def _plot_correlation_matrix(self, analysis, save_dir):
        """Plot correlation matrix between EEG and image latents"""
        if 'alignment' not in analysis or 'correlation_matrix' not in analysis['alignment']:
            return
            
        correlation = analysis['alignment']['correlation_matrix']
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, cmap='coolwarm', center=0)
        plt.title('EEG-Image Latent Space Correlation')
        plt.xlabel('Image Latent Dimensions')
        plt.ylabel('EEG Latent Dimensions')
        plt.savefig(save_dir / 'correlation_matrix.png')
        plt.close()

# =======================
# Results Manager
# =======================

def numpy_to_python(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

class ResultsManager:
    """Manages saving and loading of results"""
    def __init__(self, project_paths):
        self.paths = project_paths
        self.results_dir = project_paths.results_dir
        
    def save_session(self, session_data):
        """Save a complete processing session with proper type conversion"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.results_dir / f'session_{timestamp}'
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        serializable_data = {
            'config': session_data['config'],
            'metrics': numpy_to_python(session_data['metrics']),
            'analysis': numpy_to_python(session_data['analysis'])
        }
        
        # Save configuration
        with open(session_dir / 'config.yaml', 'w') as f:
            yaml.dump(serializable_data['config'], f)
            
        # Save metrics
        with open(session_dir / 'metrics.json', 'w') as f:
            json.dump(serializable_data['metrics'], f, indent=2)
            
        # Save generated images
        if 'images' in session_data:
            np.save(session_dir / 'generated_images.npy',
                   session_data['images'])
            
        # Save analysis results
        if 'analysis' in session_data:
            with open(session_dir / 'analysis.json', 'w') as f:
                json.dump(serializable_data['analysis'], f, indent=2)
                
        logger.info(f"Session saved to '{session_dir}'")
        return session_dir
    
    def load_session(self, session_dir):
        """Load a saved session"""
        session_dir = Path(session_dir)
        
        session_data = {}
        
        # Load configuration
        with open(session_dir / 'config.yaml', 'r') as f:
            session_data['config'] = yaml.safe_load(f)
            
        # Load metrics
        with open(session_dir / 'metrics.json', 'r') as f:
            session_data['metrics'] = json.load(f)
            
        # Load generated images
        images_path = session_dir / 'generated_images.npy'
        if images_path.exists():
            session_data['images'] = np.load(images_path)
            
        # Load analysis
        analysis_path = session_dir / 'analysis.json'
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                session_data['analysis'] = json.load(f)
                
        logger.info(f"Session loaded from '{session_dir}'")
        return session_data
    

    def list_sessions(self):
        """List all saved sessions"""
        sessions = []
        for path in self.results_dir.glob('session_*'):
            if path.is_dir():
                config_path = path / 'config.yaml'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    sessions.append({
                        'path': path,
                        'timestamp': path.name.split('_')[1],
                        'config': config
                    })
        return sessions

# =======================
# Pretrained Image Decoder
# =======================

class PretrainedImageDecoder:
    """Wrapper for the pretrained VAE decoder"""
    def __init__(self, vae_model, device='cpu'):
        self.vae = vae_model.to(device)
        self.vae.eval()
        self.device = device

    def decode(self, latents):
        with torch.no_grad():
            latents = torch.tensor(latents, dtype=torch.float32).to(self.device)
            reconstructed = self.vae.decode(latents, [])
            reconstructed = reconstructed.cpu().numpy()
            # Since decoder output is [0,1], no need for additional scaling
        return reconstructed

# =======================
# Brain Dataset
# =======================

class BrainDataset(Dataset):
    """Custom dataset for brain EEG and images"""
    def __init__(self, processed_dir, transform=None):
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        
        pairing_info_path = self.processed_dir / 'latent_pairs' / 'pairing_info.json'
        if not pairing_info_path.exists():
            raise FileNotFoundError(f"Pairing info not found at '{pairing_info_path}'")
        
        with open(pairing_info_path, 'r') as f:
            self.pairs = json.load(f)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load EEG segment
        eeg_segment = np.load(self.processed_dir / pair['eeg_stage'] / f"{pair['eeg_segment']}.npy")
        
        # Ensure consistent shape: [channels, sequence]
        if len(eeg_segment.shape) == 1:
            eeg_segment = eeg_segment[np.newaxis, :]
        elif len(eeg_segment.shape) > 2:
            # If we have additional dimensions, flatten them into the sequence dimension
            eeg_segment = eeg_segment.reshape(eeg_segment.shape[0], -1)
        
        # Convert to tensor
        eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)
        
        # Load and process image
        img_path = Path(pair['cifar_image'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return eeg_tensor, img, pair['eeg_stage']

# =======================
# Test Cases using Pytest
# =======================

class TestEEGProcessor:
    @pytest.fixture
    def eeg_processor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy EDF file or mock EEG data
            # Since creating a real EDF file is complex, we'll mock the EEGProcessor's data
            processor = EEGProcessor(Path(tmpdir) / "test.edf", window_size=64, stride=32, config={'processing': {'eeg_channels': 'all', 'min_freq': 1.0, 'max_freq': 45.0, 'normalize': True}})
            processor.data = np.random.randn(2, 1000)  # 2 channels, 1000 timepoints
            yield processor
    
    def test_window_extraction(self, eeg_processor):
        eeg_processor.load_and_preprocess = lambda: None  # Mock load_and_preprocess
        windows = eeg_processor.extract_windows()
        assert len(windows) > 0
        assert windows.shape == (29, 2, 64)  # (1000 - 64) / 32 = 29.5 => 29 windows

class TestBrainDecoder:
    @pytest.fixture
    def model(self):
        config = Config()
        config.config['model']['latent_dim'] = 256
        return BrainDecoderModel(num_channels=1, seq_length=64, config=config)
    
    def test_forward_pass(self, model):
        batch_size = 32
        num_channels = 1
        seq_length = 64
        
        eeg = torch.randn(batch_size, num_channels, seq_length)
        images = torch.randn(batch_size, 3, 32, 32)
        generated_images, eeg_features, image_features = model(eeg, images)
        
        assert generated_images.shape == (batch_size, 3, 32, 32)  # CIFAR size
    
    def test_generate(self, model):
        batch_size = 1
        num_channels = 1
        seq_length = 64
        
        eeg = torch.randn(batch_size, num_channels, seq_length)
        with torch.no_grad():
            generated = model.generate(eeg)
        
        assert generated.shape == (batch_size, 3, 32, 32)
        assert torch.all(generated >= 0) and torch.all(generated <= 1)  # Sigmoid activation

class TestResultAnalyzer:
    @pytest.fixture
    def analyzer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ProjectPaths(tmpdir)
            device = torch.device('cpu')
            yield ResultAnalyzer(paths, device)
    
    def test_metrics_computation(self, analyzer):
        n_samples = 10
        img_shape = (32, 32, 3)
        
        generated = np.random.randint(0, 255, size=(n_samples, *img_shape), dtype=np.uint8)
        target = np.random.randint(0, 255, size=(n_samples, *img_shape), dtype=np.uint8)
        
        metrics = analyzer.compute_metrics(generated, target)
        
        assert 'pixel_mse' in metrics
        assert 'ssim' in metrics
        assert 'perceptual_distance' in metrics
        assert 'feature_correlation' in metrics

def test_end_to_end():
    """Test complete pipeline"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        paths = ProjectPaths(tmpdir)
        config_instance = Config()
        config = config_instance
        device = config_instance.device
        
        # Create synthetic EEG data
        eeg_data = np.random.randn(1, 10000)  # 1 channel, 10,000 timepoints
        
        # Process EEG
        processor = EEGProcessor('dummy.edf', window_size=config.config['model']['window_size'], stride=config.config['model']['stride'], config=config)
        processor.data = eeg_data
        processor.sampling_rate = 100
        
        windows = processor.extract_windows()  # Shape: (n_samples, n_channels, window_size)
        
        # Load and scale AI latent vectors
        # For testing, we'll mock latents
        latents = np.random.randn(len(windows), config.config['model']['latent_dim'])
        labels = np.random.randint(0, 10, size=(len(windows),))
        np.save(paths.get_latent_path('cifar10_latents'), latents)
        np.save(paths.get_latent_path('cifar10_labels'), labels)
        
        # Create and run model
        model = BrainDecoderModel(
            num_channels=1,  # Adjust based on your EEG data
            seq_length=config.config['model']['window_size'],
            config=config
        ).to(device)
        
        # Initialize ImageEncoder
        image_encoder = ImageEncoder(
            latent_dim=config.config['model']['latent_dim']
        ).to(device)
        
        # Create dataset
        dataset = BrainDataset(windows, transform=transforms.ToTensor())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        
        # Initialize Simplified Contrastive Loss
        contrastive_loss = SleepPinkNoiseLoss(
            alpha_sleep=config.config['pink_noise']['alpha']
        ).to(device)
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            device=device,
            project_paths=paths,
            image_encoder=image_encoder,
            contrastive_loss=contrastive_loss,
            config=config
        )
        
        # Train for a few epochs
        trainer.train(dataloader, dataloader)
        
        # Generate images
        generated_images, _, _ = model(torch.randn(1,1,64).to(device), torch.randn(1,3,32,32).to(device))
        assert generated_images.shape == (1,3,32,32)

# =======================
# Function Definitions
# =======================

def train_and_extract_latents(config: Config, project_paths: ProjectPaths):
    """Train VAE and extract latent vectors"""
    logger.info("Starting VAE training and latent extraction...")
    # Adjusted transforms to remove normalization
    transform = transforms.Compose(config.config['model']['augmentation_transforms'] + [
        transforms.ToTensor(),
        # Removed normalization to keep [0,1] range
    ]) if config.config['model']['use_augmentation'] else transforms.Compose([
        transforms.ToTensor(),
        # Removed normalization to keep [0,1] range
    ])
    
    trainset = datasets.CIFAR10(root=str(project_paths.data_dir), train=True,
                                 download=True, transform=transform)
    val_size = int(config.config['training']['val_split'] * len(trainset))
    train_size = len(trainset) - val_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.config['model']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.config['model']['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize VAE
    vae = VAE(config).to(config.device)
    
    # Initialize TrainerVAE
    vae_trainer = TrainerVAE(model=vae, device=config.device, project_paths=project_paths, 
                             train_loader=train_loader, val_loader=val_loader, config=config)
    vae_trainer.train()
    
    # Extract latent vectors from test set
    logger.info("Extracting latent vectors from test set...")
    testset = datasets.CIFAR10(root=str(project_paths.data_dir), train=False,
                                download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=config.config['model']['batch_size'], shuffle=False, num_workers=4)
    
    vae.eval()
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels_batch in tqdm(test_loader, desc="Extracting Latents"):
            images = images.to(config.device)
            recon_images, mu, logvar = vae(images)
            latents = mu.cpu().numpy()
            all_latents.append(latents)
            all_labels.append(labels_batch.numpy())
    
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Save latent vectors and labels
    np.save(project_paths.get_latent_path('cifar10_latents'), latents)
    np.save(project_paths.get_latent_path('cifar10_labels'), labels)
    logger.info(f"Latent vectors and labels saved to '{project_paths.latents_dir}'")
    
    return latents, labels

# =======================
# Brain Decoder GUI
# =======================

class BrainDecoderGUI:
    """Graphical User Interface for Brain Decoder"""
    def __init__(self, project_paths, device, model, scaler_eeg, scaler_latent, config, results_manager, image_encoder):
        self.paths = project_paths
        self.device = device
        self.model = model
        self.scaler_eeg = scaler_eeg
        self.scaler_latent = scaler_latent
        self.config = config  # Store the config dictionary
        self.results_manager = results_manager  # Store the results manager
        self.image_encoder = image_encoder  # Image encoder for analysis
        self.image_queue = queue.Queue(maxsize=100)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Brain Decoder")
        self.root.geometry("1600x900")
        
        self.setup_gui()
        self.setup_help()
        
        # For generated images
        self.generated_images = []
        self.target_images = []
    
    def _get_or_generate_latents(self):
        """Get existing latents or generate new ones"""
        # Try to load existing latents
        latents_path = self.paths.get_latent_path('cifar10_latents')
        
        if not latents_path.exists():
            logger.info("Latent vectors not found. Generating new latents...")
            
            # Get or load VAE
            vae_path = self.paths.get_model_path('vae_best')
            if not vae_path.exists():
                # Train VAE and extract latents
                latents, _ = train_and_extract_latents(self.config, self.paths)
            else:
                # Load existing VAE and generate latents
                vae = VAE(self.config).to(self.device)
                vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                vae.eval()
                
                # Generate latents from generated images
                with torch.no_grad():
                    images_tensor = torch.tensor(self.generated_images).to(self.device).float()
                    mu, logvar, _ = vae.encode(images_tensor)
                    latents = mu.cpu().numpy()
                    
            # Save latents
            np.save(latents_path, latents)
            logger.info(f"Generated and saved latent vectors to '{latents_path}'")
        else:
            # Load existing latents
            latents = np.load(latents_path)
            logger.info(f"Loaded existing latents from '{latents_path}'")
        
        return latents

    def update_visualizations(self, analysis, session_dir):
        """Update visualizations in the analysis tab based on analysis results"""
        try:
            # Clear existing visualizations
            for widget in self.analysis_frame.winfo_children():
                widget.destroy()
                
            # Create figure with multiple subplots
            fig = Figure(figsize=(12, 8))
            
            # Latent space visualization (if available)
            if analysis.get('clustering') and analysis['clustering'].get('k_star_distribution'):
                ax1 = fig.add_subplot(221)
                k_star = analysis['clustering']['k_star_distribution']
                ax1.hist(k_star['values'], bins=50)
                ax1.axvline(k_star['mean'], color='r', linestyle='--',
                        label=f"Mean: {k_star['mean']:.3f}")
                ax1.set_title("k* Distribution")
                ax1.set_xlabel('k*')
                ax1.set_ylabel('Count')
                ax1.legend()
            
            # Correlation matrix (if available)
            if analysis.get('alignment') and analysis['alignment'].get('correlation_matrix') is not None:
                ax2 = fig.add_subplot(222)
                correlation = np.array(analysis['alignment']['correlation_matrix'])
                im = ax2.imshow(correlation, cmap='coolwarm', aspect='auto')
                ax2.set_title('EEG-Image Latent Space Correlation')
                fig.colorbar(im, ax=ax2)  # Use fig.colorbar instead of plt.colorbar
                
                # Add correlation stats if available
                if analysis['alignment'].get('mean_correlation') is not None:
                    ax2.set_xlabel(f'Mean Corr: {analysis["alignment"]["mean_correlation"]:.3f}')
                if analysis['alignment'].get('max_correlation') is not None:
                    ax2.set_ylabel(f'Max Corr: {analysis["alignment"]["max_correlation"]:.3f}')
            
            # Distribution statistics (if available)
            if analysis.get('distribution'):
                ax3 = fig.add_subplot(223)
                dist = analysis['distribution']
                if isinstance(dist.get('mean'), list):
                    ax3.plot(dist['mean'], label='Mean')
                    ax3.plot(dist['std'], label='Std')
                    ax3.set_title('Latent Distribution Statistics')
                    ax3.legend()
            
            # Sleep stage distribution (if available)
            if hasattr(self, 'sleep_stages') and self.sleep_stages:
                ax4 = fig.add_subplot(224)
                stage_counts = {}
                for stage in self.sleep_stages:
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                
                # Create bar plot
                bars = ax4.bar(stage_counts.keys(), stage_counts.values())
                ax4.set_title('Sleep Stage Distribution')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax4.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                # Rotate labels for better readability
                ax4.tick_params(axis='x', rotation=45)
            
            # Adjust layout and display
            fig.tight_layout()
            
            # Create canvas and display
            canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Switch to analysis tab
            self.notebook.select(2)  # Index 2 should be the analysis tab
            
            # Add text summary
            if analysis.get('alignment'):
                summary_text = f"Analysis Summary:\n"
                if 'n_samples' in analysis['alignment']:
                    summary_text += f"Samples analyzed: {analysis['alignment']['n_samples']}\n"
                if 'eeg_dim' in analysis['alignment']:
                    summary_text += f"EEG latent dimensions: {analysis['alignment']['eeg_dim']}\n"
                if 'image_dim' in analysis['alignment']:
                    summary_text += f"Image latent dimensions: {analysis['alignment']['image_dim']}\n"
                    
                summary_label = ttk.Label(self.analysis_frame, text=summary_text, justify=tk.LEFT)
                summary_label.pack(pady=10)
            
            self.update_status("Analysis visualizations updated")
            
        except Exception as e:
            logger.error(f"Error updating visualizations: {str(e)}", exc_info=True)
            self.update_status("Error updating visualizations")

    def _encode_generated_images(self):
        """Encode generated images to get their latent representations"""
        self.image_encoder.eval()
        with torch.no_grad():
            generated_images_tensor = torch.tensor(np.array(self.generated_images)).to(self.device).float()
            return self.image_encoder(generated_images_tensor).cpu().numpy()

    def setup_gui(self):
        """Setup main GUI components"""
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control elements
        ttk.Label(self.control_frame, text="EEG File:").pack(anchor=tk.W)
        self.file_frame = ttk.Frame(self.control_frame)
        self.file_frame.pack(fill=tk.X, pady=5)
        
        self.file_entry = ttk.Entry(self.file_frame)
        self.file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        ttk.Button(self.file_frame, text="Browse", 
                  command=self.browse_file).pack(side=tk.RIGHT)
        
        # EEG Channel Selection
        ttk.Label(self.control_frame, text="EEG Channels:").pack(anchor=tk.W, pady=(10,0))
        ttk.Label(self.control_frame, text="Enter 'all' or comma-separated channel indices (e.g., 2,5,7):").pack(anchor=tk.W)
        self.channels_var = tk.StringVar(value='all')
        ttk.Entry(self.control_frame, textvariable=self.channels_var).pack(fill=tk.X, pady=5)
        
        # Processing controls
        ttk.Label(self.control_frame, text="Processing Options:").pack(anchor=tk.W, pady=(10,0))
        
        self.window_size_var = tk.StringVar(value=str(self.config.config['model']['window_size']))
        ttk.Label(self.control_frame, text="Window Size:").pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.window_size_var).pack(fill=tk.X)
        
        self.stride_var = tk.StringVar(value=str(self.config.config['model']['stride']))
        ttk.Label(self.control_frame, text="Stride:").pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.stride_var).pack(fill=tk.X)
        
        # Action buttons
        ttk.Button(self.control_frame, text="Extract EEG Data", 
                  command=self.extract_eeg_data).pack(fill=tk.X, pady=10)
        ttk.Button(self.control_frame, text="Process EEG", 
                  command=self.process_eeg).pack(fill=tk.X, pady=10)
        ttk.Button(self.control_frame, text="Generate Video", 
                  command=self.generate_video).pack(fill=tk.X)
        ttk.Button(self.control_frame, text="Show Help", 
                  command=self.show_help).pack(fill=tk.X, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.control_frame, 
                                      variable=self.progress_var,
                                      maximum=100)
        self.progress.pack(fill=tk.X, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.control_frame, textvariable=self.status_var).pack()
        
        # Display area
        self.setup_display_area()
    
    def setup_display_area(self):
        """Setup the image display and visualization area"""
        # Notebook for different views
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # EEG visualization tab
        self.eeg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eeg_frame, text="EEG Signal")
        
        # Create matplotlib figure for EEG
        self.eeg_fig = plt.Figure(figsize=(8, 4))
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, self.eeg_frame)
        self.eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Generated images tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Generated Images")
        
        # Add controls frame at the top of image frame
        self.controls_frame = ttk.Frame(self.image_frame)
        self.controls_frame.pack(side="top", fill="x", padx=5, pady=5)
        
        # Add sleep stage selector
        ttk.Label(self.controls_frame, text="Sleep Stage:").pack(side="left", padx=5)
        self.stage_var = tk.StringVar(value="all")
        self.stage_combo = ttk.Combobox(self.controls_frame, 
                                    textvariable=self.stage_var,
                                    values=["all", "deep_sleep", "light_sleep", "rem"],
                                    state="readonly",
                                    width=15)
        self.stage_combo.pack(side="left", padx=5)
        self.stage_combo.bind('<<ComboboxSelected>>', lambda e: self.filter_images())
        
        # Add image count label
        self.count_label = ttk.Label(self.controls_frame, text="")
        self.count_label.pack(side="right", padx=5)
        
        # Scrollable Canvas for image grid
        self.canvas = tk.Canvas(self.image_frame)
        self.scrollbar = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Add pagination frame for image navigation
        self.pagination_frame = ttk.Frame(self.image_frame)
        self.pagination_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        
        # Add navigation buttons
        self.page_var = tk.IntVar(value=1)
        self.images_per_page = 16  # 4x4 grid
        
        ttk.Button(self.pagination_frame, text="<<", command=lambda: self.change_page("first")).pack(side="left", padx=2)
        ttk.Button(self.pagination_frame, text="<", command=lambda: self.change_page("prev")).pack(side="left", padx=2)
        
        # Page indicator
        self.page_label = ttk.Label(self.pagination_frame, text="Page 1")
        self.page_label.pack(side="left", padx=10)
        
        ttk.Button(self.pagination_frame, text=">", command=lambda: self.change_page("next")).pack(side="left", padx=2)
        ttk.Button(self.pagination_frame, text=">>", command=lambda: self.change_page("last")).pack(side="left", padx=2)
        
        # Counter label
        self.counter_label = ttk.Label(self.pagination_frame, text="")
        self.counter_label.pack(side="right", padx=10)
        
        # Similarity analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        # Create matplotlib figure for analysis
        self.analysis_fig = plt.Figure(figsize=(8, 4))
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, self.analysis_frame)
        self.analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def filter_images(self):
        """Filter images based on selected sleep stage"""
        if not hasattr(self, 'all_generated_images'):
            self.all_generated_images = list(zip(self.generated_images, self.target_images, self.sleep_stages))
        
        selected_stage = self.stage_var.get()
        if selected_stage == "all":
            filtered_images = self.all_generated_images
        else:
            filtered_images = [img_data for img_data in self.all_generated_images 
                            if img_data[2] == selected_stage]
        
        # Update counts
        total_count = len(filtered_images)
        self.count_label.config(text=f"Showing {total_count} images")
        
        # Reset pagination
        self.page_var.set(1)
        
        # Display filtered images
        self.display_filtered_grid([img for img, _, _ in filtered_images])

        
    def setup_help(self):
        """Setup help window content"""
        self.help_text = """
        Brain Decoder Help

        This application decodes visual information from EEG signals using deep learning.

        Steps to use:
        1. Load EEG Data:
           - Click 'Browse' to select your .edf file
           - Enter 'all' to use all EEG channels or specify comma-separated channel indices (e.g., 2,5,7)

        2. Extract EEG Data:
           - Click 'Extract EEG Data' to process and organize EEG segments based on sleep stages
           - This will also organize corresponding CIFAR images

        3. Adjust Processing Parameters:
           - Window Size: Number of EEG samples per window
           - Stride: Number of samples to advance between windows

        4. Process EEG:
           - Click 'Process EEG' to train the Brain Decoder model using the extracted EEG-image pairs

        5. View Results:
           - Navigate through the tabs to view EEG signals, generated images, and analysis results respectively

        6. Generate Video:
           - Click 'Generate Video' to create a video from the generated images
           - Video is saved in the results directory

        Notes:
        - Processing may take several minutes depending on data size and hardware
        - Generated images are interpretive reconstructions
        - Quality depends on EEG signal quality and selected channels
        - Ensure that EEG data extraction is completed before processing EEG
        """

    def show_help(self):
        """Display help window"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Brain Decoder Help")
        help_window.geometry("600x400")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, self.help_text)
        text_widget.config(state=tk.DISABLED)
    
    def browse_file(self):
        """Open file dialog to select EEG file"""
        filename = filedialog.askopenfilename(
            initialdir=str(self.paths.eeg_dir),
            title="Select EEG File",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    def update_status(self, message):
        """Update status message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def plot_eeg(self, eeg_data):
        """Plot EEG signal in the EEG tab"""
        self.eeg_fig.clear()
        ax = self.eeg_fig.add_subplot(111)
        ax.plot(eeg_data.T)
        channels = self.config.config['processing']['eeg_channels']
        if channels == 'all':
            channel_info = "All Channels"
        else:
            channel_info = f"Channels: {', '.join(map(str, channels))}"
        ax.set_title(f'EEG Signal ({channel_info})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        self.eeg_canvas.draw()

    def change_page(self, action):
        """Handle page navigation"""
        if not hasattr(self, 'all_generated_images'):
            return
                
        # Get current filtered images based on sleep stage selection
        selected_stage = self.stage_var.get()
        if selected_stage == "all":
            filtered_images = self.all_generated_images
        else:
            filtered_images = [img_data for img_data in self.all_generated_images 
                            if img_data[2] == selected_stage]
        
        # Calculate total pages
        total_pages = (len(filtered_images) + self.images_per_page - 1) // self.images_per_page
        current_page = self.page_var.get()
        
        # Handle different navigation actions
        if action == "next" and current_page < total_pages:
            self.page_var.set(current_page + 1)
        elif action == "prev" and current_page > 1:
            self.page_var.set(current_page - 1)
        elif action == "first":
            self.page_var.set(1)
        elif action == "last":
            self.page_var.set(total_pages)
        
        # Update display
        self.display_filtered_grid([img for img, _, _ in filtered_images])
        
        # Update page indicator
        self.page_label.config(text=f"Page {self.page_var.get()} of {total_pages}")
        self.counter_label.config(text=f"Total Images: {len(filtered_images)}")
    
    def display_filtered_grid(self, filtered_images, rows=4, cols=4):
        """Display grid of filtered images with pagination"""
        # Debug logging
        logger.info(f"Total images: {len(filtered_images)}")  # Changed from images to filtered_images

        # Clear existing grid
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if not filtered_images:  # Changed from images to filtered_images
            logger.warning("No images to display")
            label = ttk.Label(self.scrollable_frame, text="No images generated yet")
            label.pack(pady=20)
            return

        # Calculate pagination
        self.images_per_page = rows * cols
        current_page = self.page_var.get()
        total_pages = (len(filtered_images) + self.images_per_page - 1) // self.images_per_page
        
        # Update page display
        self.page_label.config(text=f"Page {current_page} of {total_pages}")
        self.counter_label.config(text=f"Total Images: {len(filtered_images)}")
        
        # Calculate start and end indices for current page
        start_idx = (current_page - 1) * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(filtered_images))
        
        # Create grid for current page
        for i, img in enumerate(filtered_images[start_idx:end_idx]):  # Changed from images to filtered_images
            frame = ttk.Frame(self.scrollable_frame)
            frame.grid(row=i//cols, column=i%cols, padx=5, pady=5)

            try:
                # Image index label
                idx_label = ttk.Label(frame, text=f"Image {start_idx + i + 1}")
                idx_label.pack()

                # Convert to PhotoImage
                if img.shape[0] == 3:  # Channel first format (3, H, W)
                    img_np = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                elif img.shape[0] == 1:  # Single channel
                    img_np = (img.squeeze(0) * 255).clip(0, 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np, mode='L')
                else:
                    logger.warning(f"Unexpected image shape: {img.shape}")
                    continue

                img_pil = img_pil.resize((100, 100))
                img_tk = ImageTk.PhotoImage(img_pil)

                # Display image
                label = ttk.Label(frame, image=img_tk)
                label.image = img_tk  # Keep reference
                label.pack()

            except Exception as e:
                logger.error(f"Error displaying image {start_idx + i}: {str(e)}")
                error_label = ttk.Label(frame, text=f"Error: {str(e)}")
                error_label.pack()

        # Force update
        self.scrollable_frame.update_idletasks()

    
    def plot_psd(self, latent_vectors, title, save_path):
        """Plot Power Spectral Density of latent vectors"""
        latent = latent_vectors.cpu().numpy()
        psd = []
        for i in range(latent.shape[1]):
            freq, power = welch(latent[:, i], fs=256, nperseg=min(256, latent.shape[1]))
            psd.append(power)
        psd = np.array(psd)
        mean_psd = np.mean(psd[:, 1:], axis=0)
        plt.figure(figsize=(10, 6))
        plt.loglog(freq[1:], mean_psd[1:], label='Latent PSD')
        plt.loglog(freq[1:], 1 / (freq[1:] ** self.config.config['pink_noise']['alpha']), label='1/f^alpha', linestyle='--')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectral Density')
        plt.title(title)
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def generate_images_from_eeg(self):
        """Generate images using pretrained models without training"""
        try:
            self.update_status("Generating images from EEG using pretrained models...")
            
            # Load dataset
            processed_dir = self.paths.processed_dir
            dataset = BrainDataset(processed_dir, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]))
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            
            # Generate images
            self.model.eval()  # Ensure model is in eval mode
            with torch.no_grad():  # No gradients needed for inference
                for eeg_batch, image_batch, _ in tqdm(dataloader, desc="Generating Images"):
                    if len(eeg_batch.shape) == 3:
                        eeg_batch = eeg_batch.unsqueeze(1)
                    generated_images, _, _ = self.model(eeg_batch.to(self.device), image_batch.to(self.device))
                    generated_images = generated_images.cpu().numpy()
                    self.generated_images.extend(generated_images)
                    self.target_images.extend(image_batch.numpy())
            
            # Display results
            self.display_image_grid(self.generated_images)
            self.notebook.select(1)  # Switch to Generated Images tab
            self.update_status("Images generated successfully")
            
            # Perform analysis
            self.perform_analysis()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate images: {str(e)}")
            logger.error(f"Image generation error: {str(e)}", exc_info=True)
    
    def perform_analysis(self):
        """Perform analysis after processing"""
        if not self.generated_images:
            return

        try:
            # Use sleep_stages as labels
            labels = np.array(self.sleep_stages)
            
            # Get latents
            latents = self._get_or_generate_latents()
            
            # Ensure we have the right number of latents
            if len(latents) != len(labels):
                logger.warning(f"Number of latents ({len(latents)}) does not match number of labels ({len(labels)})")
                # Adjust latents or labels as needed
                min_len = min(len(latents), len(labels))
                latents = latents[:min_len]
                labels = labels[:min_len]
            
            # Scale latents
            scaled_latents = self.scaler_latent.fit_transform(latents)
            
            # Get generated latent representations
            generated_latents = self._encode_generated_images()
            
            # Analyze results
            analyzer = ResultAnalyzer(self.paths, self.device)
            metrics = analyzer.compute_metrics(self.generated_images, self.target_images)
            
            # Check unique labels before calling analyze_latent_space
            if len(np.unique(labels)) > 1:
                analysis = analyzer.analyze_latent_space(scaled_latents, generated_latents, labels)
            else:
                logger.warning("Only one unique label found - skipping clustering analysis")
                analysis = analyzer.analyze_latent_space(scaled_latents, generated_latents, None)
            
            # Save session
            session_data = {
                'config': self.config.config,
                'metrics': metrics,
                'images': self.generated_images,
                'analysis': analysis
            }
            session_dir = self.results_manager.save_session(session_data)
            
            # Schedule visualization update in main thread
            if any(v is not None for v in analysis.values()):
                self.root.after(100, lambda: self.update_visualizations(analysis, session_dir))
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            self.update_status(f"Analysis partially completed with errors: {str(e)}")

    def generate_video(self):
        try:
            self.update_status("Generating video...")
            
            # If we don't have generated images yet, but we have trained models, generate them
            if len(self.generated_images) == 0:
                # Check if we have trained models
                brain_decoder_path = self.paths.get_model_path('brain_decoder_best')
                image_encoder_path = self.paths.get_model_path('image_encoder_best')
                modality_alignment_path = self.paths.get_model_path('modality_alignment_best')
                if brain_decoder_path.exists() and image_encoder_path.exists() and modality_alignment_path.exists():
                    # Load saved models
                    self.model.load_state_dict(torch.load(brain_decoder_path, map_location=self.device))
                    self.image_encoder.load_state_dict(torch.load(image_encoder_path, map_location=self.device))
                    self.model.modality_alignment.load_state_dict(torch.load(modality_alignment_path, map_location=self.device))
                    self.model.eval()
                    self.image_encoder.eval()
                    
                    # Get test data and generate images
                    processed_dir = self.paths.data_dir / 'processed'
                    dataset = BrainDataset(processed_dir, transform=transforms.ToTensor())
                    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
                    
                    with torch.no_grad():
                        for eeg_batch, image_batch, _ in test_loader:
                            if len(eeg_batch.shape) == 3:
                                eeg_batch = eeg_batch.unsqueeze(1)
                            generated_images, _, _ = self.model(eeg_batch.to(self.device), image_batch.to(self.device))
                            self.generated_images = list(generated_images.cpu().numpy())  # Ensure it's a list
                            self.target_images = list(image_batch.numpy())
                            # Display the generated images
                            self.display_image_grid(self.generated_images)
                            self.notebook.select(1)  # Switch to Generated Images tab
                            break  # Just get one batch for the video
                        else:
                            self.update_status("No trained models found. Please process EEG data first.")
                            return
                else:
                    self.update_status("No trained models found. Please process EEG data first.")
                    return
            
            if len(self.generated_images) == 0:
                self.update_status("No images available for video")
                return
                
            # Create video
            output_path = self.paths.results_dir / 'brain_decoding.mp4'
            
            # Convert images from (3,32,32) to (32,32,3)
            images_rgb = [img.transpose(1,2,0) for img in self.generated_images]
            
            # Write video using imageio
            imageio.mimsave(str(output_path), images_rgb, fps=self.config.config['visualization']['fps'])
            
            self.update_status(f"Video saved to {output_path}")
            
        except Exception as e:
            self.update_status(f"Error generating video: {str(e)}")
            logger.error(f"Video generation error: {str(e)}", exc_info=True)
    
    def extract_eeg_data(self):
        """Extract and organize EEG data segments"""
        edf_file = self.file_entry.get()
        channels = self.channels_var.get()
        
        if not edf_file:
            messagebox.showerror("Error", "Please select an EDF file.")
            return
            
        try:
            self.update_status("Extracting EEG data...")
            extractor = EEGExtractor(edf_file, selected_channels=channels)
            extractor.load_edf()
            
            save_dir = self.paths.data_dir / 'processed'
            extractor.extract_segments(save_dir)
            
            # Create latent pairs if VAE is available
            vae_path = self.paths.get_model_path('vae_best')
            if vae_path.exists():
                vae = VAE(self.config).to(self.device)
                vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                vae.eval()
                extractor.create_latent_pairs(save_dir, vae)
            else:
                logger.warning(f"VAE model not found at '{vae_path}'. Latent pairs will not be created.")
            
            self.update_status("EEG data extracted and organized successfully")
            
            # Update visualization
            self.plot_stage_distribution(save_dir)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract EEG data: {str(e)}")
            logger.error(f"EEG extraction error: {str(e)}", exc_info=True)
    
        # Plot EEG data
        try:
            data = extractor.raw.get_data()
            self.plot_eeg(data)
        except Exception as e:
            logger.error(f"Error plotting EEG data: {str(e)}", exc_info=True)
    
    def plot_stage_distribution(self, processed_dir):
        """Plot distribution of sleep stages"""
        with open(processed_dir / 'eeg_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        stages = metadata['stages']
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.bar(stages.keys(), stages.values())
        ax.set_title('Sleep Stage Distribution')
        ax.set_ylabel('Number of Segments')
        
        # Update analysis tab
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def process_eeg(self):
        """Process the extracted EEG segments"""
        processed_dir = self.paths.data_dir / 'processed'
        if not processed_dir.exists():
            messagebox.showerror("Error", "Please extract EEG data first.")
            return
            
        try:
            # Load pairing info
            pairing_info_path = processed_dir / 'latent_pairs' / 'pairing_info.json'
            if not pairing_info_path.exists():
                messagebox.showerror("Error", "Pairing info not found. Please ensure EEG data extraction includes latent pairing.")
                return
            
            with open(pairing_info_path, 'r') as f:
                pairing_info = json.load(f)
            
            # Load dataset
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            dataset = BrainDataset(processed_dir, transform=transform)
            
            # Split dataset
            train_size = int(self.config.config['training']['train_split'] * len(dataset))
            val_size = int(self.config.config['training']['val_split'] * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.config.config['model']['batch_size'], 
                                    shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=self.config.config['model']['batch_size'], 
                                shuffle=False, num_workers=4)
            
            # Load best models if they exist
            brain_decoder_path = self.paths.get_model_path('brain_decoder_best')
            image_encoder_path = self.paths.get_model_path('image_encoder_best')
            modality_alignment_path = self.paths.get_model_path('modality_alignment_best')
            
            if brain_decoder_path.exists() and image_encoder_path.exists() and modality_alignment_path.exists():
                user_choice = messagebox.askyesnocancel(
                    "Pretrained Models Found",
                    "Pretrained models found. Would you like to:\n\n" +
                    "• Click 'Yes' to use existing models (no training)\n" +
                    "• Click 'No' to train new models from scratch\n" +
                    "• Click 'Cancel' to exit",
                    icon='question'
                )
                
                if user_choice is None:  # Cancel
                    return
                
                if user_choice:  # Yes - use existing models
                    # Load saved models
                    self.model.load_state_dict(torch.load(brain_decoder_path, map_location=self.device))
                    self.image_encoder.load_state_dict(torch.load(image_encoder_path, map_location=self.device))
                    self.model.modality_alignment.load_state_dict(torch.load(modality_alignment_path, map_location=self.device))
                    self.model.eval()
                    self.image_encoder.eval()
                    logger.info("Loaded pretrained models for inference")
                    
                    # Generate images for ALL data, not just test set
                    logger.info("Generating images using pretrained models...")
                    
                    # Use full dataset for generation
                    full_loader = DataLoader(dataset, batch_size=self.config.config['model']['batch_size'], 
                                        shuffle=False, num_workers=4)
                    
                    self.generated_images = []
                    self.target_images = []
                    self.sleep_stages = []  # Add storage for sleep stages
                    
                    with torch.no_grad():
                        for eeg_batch, image_batch, stages in tqdm(full_loader, desc="Generating Images"):
                            if len(eeg_batch.shape) == 3:  # If shape is (batch, channels, time)
                                eeg_batch = eeg_batch.unsqueeze(1)  # Add extra dimension if needed
                            
                            generated_images, _, _ = self.model(eeg_batch.to(self.device), image_batch.to(self.device))
                            self.generated_images.extend(generated_images.cpu().numpy())
                            self.target_images.extend(image_batch.numpy())
                            self.sleep_stages.extend(stages)  # Store the sleep stages
                            
                            # Optional: Log progress
                            if len(self.generated_images) % 500 == 0:
                                logger.info(f"Generated {len(self.generated_images)} images so far")
                    
                    # Store all data together for filtering
                    self.all_generated_images = list(zip(self.generated_images, 
                                                    self.target_images, 
                                                    self.sleep_stages))
                    
                    # Log stage distribution
                    stage_counts = {}
                    for stage in self.sleep_stages:
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    logger.info(f"Generated images by sleep stage: {stage_counts}")
                    
                    # Display initial results (all stages)
                    self.filter_images()  # This will trigger the display with "all" stages
                    self.notebook.select(1)  # Switch to Generated Images tab
                    
                    # Perform analysis
                    self.perform_analysis()
                    
                    self.update_status("Processing completed successfully.")
                    return
                
                else:  # No - train new models
                    # Initialize fresh models
                    self.model = BrainDecoderModel(
                        num_channels=1,
                        seq_length=self.config.config['model']['window_size'],
                        config=self.config
                    ).to(self.device)
                    self.image_encoder = ImageEncoder(
                        latent_dim=self.config.config['model']['latent_dim']
                    ).to(self.device)
                    logger.info("Initialized new models for training")
            
            # Initialize Simplified Contrastive Loss
            contrastive_loss = SleepPinkNoiseLoss(
                alpha_sleep=self.config.config['pink_noise']['alpha']
            ).to(self.device)
            
            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                device=self.device,
                project_paths=self.paths,
                image_encoder=self.image_encoder,
                contrastive_loss=contrastive_loss,
                config=self.config
            )
            
            # Start training in a separate thread to keep GUI responsive
            threading.Thread(
                target=self.run_training,
                args=(trainer, train_loader, val_loader),
                daemon=True
            ).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process EEG data: {str(e)}")
            logger.error(f"EEG processing error: {str(e)}", exc_info=True)
            
    def run_training(self, trainer, train_loader, val_loader):
        """Run the training loop"""
        try:
            self.update_status("Starting training of BrainDecoderModel...")
            trainer.train(train_loader, val_loader)
            self.update_status("Training completed. Generating images...")
            
            # Generate images for test set
            test_eeg, test_images, _ = next(iter(train_loader))
            
            # Ensure proper dimensions for EEG data
            if len(test_eeg.shape) == 3:  # If shape is (batch, channels, time)
                test_eeg = test_eeg.unsqueeze(1)  # Add extra dimension if needed
            
            with torch.no_grad():
                generated_images, _, _ = self.model(test_eeg.to(self.device), test_images.to(self.device))
                generated_images = generated_images.cpu().numpy()
                logger.info(f"Generated {len(generated_images)} images with shape {generated_images.shape}")
                
                self.generated_images = list(generated_images)  # Ensure it's a list
                self.target_images = list(test_images.numpy())
                
                logger.info(f"Stored {len(self.generated_images)} images")
            
            # Display generated images and switch to images tab
            def update_display():
                self.display_image_grid(self.generated_images)
                self.notebook.select(1)  # Select Generated Images tab
                logger.info("Updated display and switched to images tab")
                
            self.root.after(100, update_display)  # Small delay to ensure GUI is ready
            
            # Perform analysis
            self.perform_analysis()
            
            self.update_status("Processing completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")
            logger.error(f"Training error: {str(e)}", exc_info=True)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

# =======================
# Test Cases using Pytest
# =======================

# (Test cases are defined above in the Test Classes section)

# =======================
# Function Definitions
# =======================

def train_and_extract_latents(config: Config, project_paths: ProjectPaths):
    """Train VAE and extract latent vectors"""
    logger.info("Starting VAE training and latent extraction...")
    # Adjusted transforms to remove normalization
    transform = transforms.Compose(config.config['model']['augmentation_transforms'] + [
        transforms.ToTensor(),
        # Removed normalization to keep [0,1] range
    ]) if config.config['model']['use_augmentation'] else transforms.Compose([
        transforms.ToTensor(),
        # Removed normalization to keep [0,1] range
    ])
    
    trainset = datasets.CIFAR10(root=str(project_paths.data_dir), train=True,
                                 download=True, transform=transform)
    val_size = int(config.config['training']['val_split'] * len(trainset))
    train_size = len(trainset) - val_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.config['model']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.config['model']['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize VAE
    vae = VAE(config).to(config.device)
    
    # Initialize TrainerVAE
    vae_trainer = TrainerVAE(model=vae, device=config.device, project_paths=project_paths, 
                             train_loader=train_loader, val_loader=val_loader, config=config)
    vae_trainer.train()
    
    # Extract latent vectors from test set
    logger.info("Extracting latent vectors from test set...")
    testset = datasets.CIFAR10(root=str(project_paths.data_dir), train=False,
                                download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=config.config['model']['batch_size'], shuffle=False, num_workers=4)
    
    vae.eval()
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels_batch in tqdm(test_loader, desc="Extracting Latents"):
            images = images.to(config.device)
            recon_images, mu, logvar = vae(images)
            latents = mu.cpu().numpy()
            all_latents.append(latents)
            all_labels.append(labels_batch.numpy())
    
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Save latent vectors and labels
    np.save(project_paths.get_latent_path('cifar10_latents'), latents)
    np.save(project_paths.get_latent_path('cifar10_labels'), labels)
    logger.info(f"Latent vectors and labels saved to '{project_paths.latents_dir}'")
    
    return latents, labels

# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    import argparse
    import torchvision  # Required for TrainerVAE's save_reconstructions
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Brain Decoder Application")
    parser.add_argument('--config', type=str, default=None, help='Path to configuration YAML file')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare CIFAR dataset before first use')
    args = parser.parse_args()
    
    # Load configuration
    config_instance = Config(config_path=args.config)
    config = config_instance
    device = config_instance.device  # Access device from config_instance
    
    # Setup project paths
    project_paths = ProjectPaths()
    
    # Prepare CIFAR-10 dataset if requested
    if args.prepare_data:
        logger.info("Preparing CIFAR-10 dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Removed normalization to keep [0,1] range
        ])
        trainset = datasets.CIFAR10(root=str(project_paths.data_dir), train=True,
                                     download=True, transform=transform)
        testset = datasets.CIFAR10(root=str(project_paths.data_dir), train=False,
                                    download=True, transform=transform)
        logger.info("CIFAR-10 dataset prepared.")
        sys.exit(0)
    
    # Check if VAE is already trained
    vae_path = project_paths.get_model_path('vae_best')
    if vae_path.exists():
        logger.info(f"Loading pretrained VAE from '{vae_path}'")
        vae = VAE(config).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
    else:
        # Train VAE and extract latents
        latents, labels = train_and_extract_latents(config, project_paths)
    
    # Initialize scalers
    scaler_eeg = StandardScaler()
    scaler_latent = StandardScaler()
    
    # Load VAE model
    # Note: If VAE was loaded above, no need to train again
    if not vae_path.exists():
        # VAE has been trained in train_and_extract_latents
        # Save the trained VAE
        torch.save(vae.state_dict(), vae_path)
        logger.info(f"Saved trained VAE model to '{vae_path}'")
    else:
        # VAE was loaded, no action needed
        pass
    
    # Initialize Pretrained Image Decoder
    image_decoder = PretrainedImageDecoder(vae, device=device)
    
    # Initialize BrainDecoderModel
    brain_decoder_model = BrainDecoderModel(
        num_channels=1,  # Adjust based on your EEG data
        seq_length=config.config['model']['window_size'],
        config=config
    ).to(device)
    
    # Initialize ImageEncoder
    image_encoder = ImageEncoder(latent_dim=config.config['model']['latent_dim']).to(device)
    
    # Load ImageEncoder model if exists
    image_encoder_path = project_paths.get_model_path('image_encoder_best')
    if image_encoder_path.exists():
        image_encoder.load_state_dict(torch.load(image_encoder_path, map_location=device))
        image_encoder.eval()
    else:
        logger.warning(f"ImageEncoder model not found at '{image_encoder_path}'. Analysis may be limited.")
    
    # Initialize Simplified Contrastive Loss
    contrastive_loss = SleepPinkNoiseLoss(alpha_sleep=config.config['pink_noise']['alpha']).to(device)
    
    # Initialize Trainer
    # Check if BrainDecoderModel is already trained
    brain_decoder_model_path = project_paths.get_model_path('brain_decoder_best')
    if not brain_decoder_model_path.exists():
        logger.info("BrainDecoderModel not found. It will be trained when you process EEG data via the GUI.")
    else:
        logger.info(f"Loading pretrained BrainDecoderModel from '{brain_decoder_model_path}'")
        brain_decoder_model.load_state_dict(torch.load(brain_decoder_model_path, map_location=device))
        brain_decoder_model.eval()
    
    # Initialize Results Manager
    results_manager = ResultsManager(project_paths)
    
    # Initialize and run GUI
    try:
        gui = BrainDecoderGUI(
            project_paths=project_paths,
            device=device,
            model=brain_decoder_model,
            scaler_eeg=scaler_eeg,
            scaler_latent=scaler_latent,
            config=config,
            results_manager=results_manager,
            image_encoder=image_encoder
        )
        
        gui.run()
    except Exception as e:
        logger.error(f"Failed to start GUI: {str(e)}", exc_info=True)
        sys.exit(1)
