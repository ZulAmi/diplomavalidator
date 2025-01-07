from tensorflow import tf, VGG16, Model, Dense, GlobalAveragePooling2D, Dropout, Adam, EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

class DiplomaClassifier:
    def __init__(self, input_shape=(600, 800, 1), model_path=None):
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = self._build_model()
        
    def _build_model(self):
        # Base model (VGG16)
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.input_shape[0], self.input_shape[1], 3)
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Load existing weights if provided
        if self.model_path and os.path.exists(self.model_path):
            model.load_weights(self.model_path)
            
        return model
    
    def train(self, training_data, labels, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model with the provided data"""
        
        # Convert grayscale to RGB for VGG16
        if training_data.shape[-1] == 1:
            training_data = np.repeat(training_data, 3, axis=-1)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                save_weights_only=True
            )
        ]
        
        # Train
        history = self.model.fit(
            training_data,
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        """Predict whether an image is a real or fake diploma"""
        
        # Ensure image has correct dimensions
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Make prediction
        prediction = self.model.predict(image)
        
        return bool(prediction[0] > 0.5)  # True for real, False for fake
    
    def evaluate(self, test_data, test_labels):
        """Evaluate model performance on test data"""
        
        # Convert grayscale to RGB for VGG16
        if test_data.shape[-1] == 1:
            test_data = np.repeat(test_data, 3, axis=-1)
            
        # Get predictions
        predictions = self.model.predict(test_data)
        predictions_binary = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(test_labels, predictions_binary),
            'confusion_matrix': confusion_matrix(test_labels, predictions_binary),
            'accuracy': self.model.evaluate(test_data, test_labels)[1]
        }
        
        return metrics
    
    def save_model(self, path):
        """Save model weights"""
        self.model.save_weights(path)
        
    def load_model(self, path):
        """Load model weights"""
        self.model.load_weights(path)