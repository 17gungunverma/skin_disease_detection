
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, Sequence
from config import IMG_SIZE, VALIDATION_SPLIT, RANDOM_SEED

# Always normalise IMG_SIZE to a (H, W) tuple
# config.py may define it as an int (224) or a tuple ((224, 224))
_IMG_SIZE = (IMG_SIZE, IMG_SIZE) if isinstance(IMG_SIZE, int) else tuple(IMG_SIZE)


# ──────────────────────────────────────────────────────────────────────────────
# Keras Sequence Generator  (loads one batch at a time → no RAM overflow)
# ──────────────────────────────────────────────────────────────────────────────
class HAM10000Generator(Sequence):
    """Batch generator – reads images from disk on-the-fly."""

    def __init__(self, image_paths, labels, batch_size=32,
                 img_size=_IMG_SIZE, augment=False):
        self.image_paths = image_paths
        self.labels      = labels
        self.batch_size  = batch_size
        # Always store as (H, W) tuple
        self.img_size    = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.augment     = augment
        self.indices     = np.arange(len(self.image_paths))

    # number of batches per epoch
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    # return one batch
    def __getitem__(self, idx):
        batch_idx    = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths  = [self.image_paths[i] for i in batch_idx]
        batch_labels = [self.labels[i]      for i in batch_idx]

        H, W = self.img_size  # unpack once, clearly

        images = []
        for path in batch_paths:
            img = cv2.imread(path)
            if img is None:
                # fallback: black image
                img = np.zeros((H, W, 3), dtype=np.uint8)
            img = cv2.resize(img, (W, H))          # cv2.resize takes (W, H)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            images.append(img)

        return np.array(images, dtype='float32'), np.array(batch_labels, dtype='float32')

    # shuffle at end of every epoch
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ──────────────────────────────────────────────────────────────────────────────
# Main Loader
# ──────────────────────────────────────────────────────────────────────────────
class HAM10000DataLoader:
    """Loads and preprocesses HAM10000 dataset using memory-efficient generators."""

    DISEASE_MAPPING = {
        'akiec': 'Actinic keratosis',
        'bcc'  : 'Basal cell carcinoma',
        'bkl'  : 'Benign keratosis',
        'df'   : 'Dermatofibroma',
        'mel'  : 'Melanoma',
        'nv'   : 'Melanocytic nevus',
        'vasc' : 'Vascular lesion'
    }

    def __init__(self, data_dir='data/',
                 metadata_file='data/HAM10000_metadata.csv'):
        self.data_dir      = data_dir
        self.metadata_file = metadata_file
        self.images_dir    = os.path.join(data_dir, 'HAM10000_images')
        self.label_encoder = LabelEncoder()
        self.class_weights = {}

        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.metadata_file):
            raise ValueError(f"Metadata file not found: {self.metadata_file}")

    # ──────────────────────────────────────────────────────────────────────────
    def load_data(self, test_size=0.2, batch_size=32):
        """
        Build train / validation generators.

        Returns:
            train_gen  : HAM10000Generator  for training
            val_gen    : HAM10000Generator  for validation
            le         : fitted LabelEncoder (needed to decode predictions)
        """
        print("[v0] Loading HAM10000 dataset...")
        print(f"[v0] Images directory : {self.images_dir}")
        print(f"[v0] Metadata file    : {self.metadata_file}")

        # ── load metadata ────────────────────────────────────────────────────
        df = pd.read_csv(self.metadata_file)
        print(f"[v0] Loaded metadata for {len(df)} images")

        # normalise dx to lowercase so mapping always works
        df['dx'] = df['dx'].str.lower().str.strip()

        # ── build full image paths ────────────────────────────────────────────
        df['path'] = df['image_id'].apply(
            lambda x: os.path.join(self.images_dir, str(x) + '.jpg')
        )

        # drop rows whose image file is missing
        before = len(df)
        df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
        missing = before - len(df)
        if missing:
            print(f"[v0] Warning: Skipped {missing} rows – image file not found")

        print(f"[v0] Using {len(df)} images for training/validation")

        # ── encode labels ─────────────────────────────────────────────────────
        y_encoded = self.label_encoder.fit_transform(df['dx'])
        num_classes = len(self.label_encoder.classes_)
        y_onehot = to_categorical(y_encoded, num_classes=num_classes)

        # ── class distribution ────────────────────────────────────────────────
        print("\n[v0] Class Distribution:")
        unique, counts = np.unique(y_encoded, return_counts=True)
        for cls_idx, cnt in zip(unique, counts):
            cls_name = self.DISEASE_MAPPING.get(
                self.label_encoder.classes_[cls_idx], 
                self.label_encoder.classes_[cls_idx]
            )
            pct = cnt / len(y_encoded) * 100
            print(f"  {cls_name:30s}: {cnt:5d}  ({pct:.1f}%)")

        # ── train / val split ────────────────────────────────────────────────
        paths = df['path'].tolist()

        (paths_train, paths_val,
         y_train,     y_val,
         ye_train,    _) = train_test_split(
            paths, y_onehot, y_encoded,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y_encoded
        )

        print(f"\n[v0] Data split:")
        print(f"  Training   : {len(paths_train)} images")
        print(f"  Validation : {len(paths_val)}   images")

        # ── class weights ─────────────────────────────────────────────────────
        self.class_weights = self._calculate_class_weights(ye_train, num_classes)

        # ── build generators ──────────────────────────────────────────────────
        train_gen = HAM10000Generator(paths_train, y_train,
                                      batch_size=batch_size, augment=True)
        val_gen   = HAM10000Generator(paths_val,   y_val,
                                      batch_size=batch_size, augment=False)

        return train_gen, val_gen, self.label_encoder

    # ──────────────────────────────────────────────────────────────────────────
    def _calculate_class_weights(self, y_encoded, num_classes):
        """Return dict of class weights to handle class imbalance."""
        unique, counts = np.unique(y_encoded, return_counts=True)
        total = len(y_encoded)

        class_weights = {}
        print("\n[v0] Class Weights (for handling imbalance):")
        for cls_idx, cnt in zip(unique, counts):
            weight = total / (num_classes * cnt)
            class_weights[int(cls_idx)] = weight
            cls_name = self.DISEASE_MAPPING.get(
                self.label_encoder.classes_[cls_idx],
                self.label_encoder.classes_[cls_idx]
            )
            print(f"  {cls_name:30s}: {weight:.4f}")

        return class_weights

    # ──────────────────────────────────────────────────────────────────────────
    def get_class_weights(self):
        return self.class_weights

    def get_disease_mapping(self):
        return self.DISEASE_MAPPING

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def verify_dataset(data_dir='data/'):
        """Verify HAM10000 dataset is properly downloaded and extracted."""
        results = {
            'metadata_exists'   : False,
            'images_dir_exists' : False,
            'image_count'       : 0,
            'metadata_count'    : 0,
            'status'            : 'Not verified'
        }

        metadata_file = os.path.join(data_dir, 'HAM10000_metadata.csv')
        if os.path.exists(metadata_file):
            results['metadata_exists'] = True
            results['metadata_count']  = len(pd.read_csv(metadata_file))

        images_dir = os.path.join(data_dir, 'HAM10000_images')
        if os.path.exists(images_dir):
            results['images_dir_exists'] = True
            results['image_count'] = sum(
                1 for f in os.listdir(images_dir) if f.endswith('.jpg')
            )

        if results['metadata_exists'] and results['images_dir_exists']:
            if results['image_count'] >= 10000 and results['metadata_count'] >= 10000:
                results['status'] = 'Complete'
            else:
                results['status'] = (
                    f"Incomplete "
                    f"(Images: {results['image_count']}, "
                    f"Metadata: {results['metadata_count']})"
                )
        else:
            results['status'] = 'Missing dataset files'

        return results