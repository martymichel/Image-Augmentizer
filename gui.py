from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QGroupBox,
    QFileDialog, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt
from typing import List, Optional
from pathlib import Path
from augmentation_types import AugmentationType, AugmentationConfig
from image_processor import ImageProcessor

class AugmentationSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Augmentation Selector")
        self.setMinimumWidth(600)
        self.setMinimumHeight(600)  
        self.input_dir = None
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Directory selection
        dir_group = QGroupBox("Input Directory")
        dir_layout = QHBoxLayout(dir_group)
        self.dir_label = QLabel("No directory selected")
        dir_layout.addWidget(self.dir_label)
        dir_btn = QPushButton("Select Directory")
        dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(dir_btn)
        layout.addWidget(dir_group)
        
        # Selection area
        self.selection_boxes: List[QComboBox] = []
        selection_group = QGroupBox("Select exactly 3 augmentation types")
        selection_layout = QVBoxLayout(selection_group)
        
        for i in range(3):
            combo = QComboBox()
            combo.addItem("Select augmentation...")
            for aug_type in AugmentationType:
                combo.addItem(str(aug_type))
            combo.currentIndexChanged.connect(self.validate_selections)
            self.selection_boxes.append(combo)
            selection_layout.addWidget(combo)
        
        layout.addWidget(selection_group)
        
        # Intensity settings
        self.intensity_widgets: List[Optional[tuple[QSpinBox, QSpinBox]]] = [None] * 3
        self.intensity_group = QGroupBox("Set intensity ranges (%)")
        self.intensity_layout = QVBoxLayout(self.intensity_group)
        layout.addWidget(self.intensity_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Summary text
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setVisible(False)
        layout.addWidget(self.summary)
        
        # Process button and status
        button_layout = QHBoxLayout()
        self.status_label = QLabel("")
        button_layout.addWidget(self.status_label)
        self.process_btn = QPushButton("Start Augmentation")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_images)
        button_layout.addWidget(self.process_btn)
        layout.addLayout(button_layout)
        
        self.show()
    
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Input Directory", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if dir_path:
            self.input_dir = Path(dir_path)
            self.dir_label.setText(str(self.input_dir))
            self.validate_selections()
    
    def validate_selections(self):
        # Get selected items (excluding the placeholder)
        selections = [
            box.currentText() for box in self.selection_boxes 
            if box.currentIndex() > 0
        ]
        
        # Check for duplicates
        if len(selections) != len(set(selections)):
            self.process_btn.setEnabled(False)
            return
        
        # Update intensity settings
        self.update_intensity_settings()
        
        # Enable process button if exactly 3 augmentations are selected
        # and directory is selected
        self.process_btn.setEnabled(
            len(selections) == 3 and self.input_dir is not None
        )
    
    def update_intensity_settings(self):
        # Clear existing intensity settings
        while self.intensity_layout.count():
            item = self.intensity_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add new intensity settings for selected augmentations
        for i, box in enumerate(self.selection_boxes):
            if box.currentIndex() > 0:
                aug_name = box.currentText()
                
                group = QWidget()
                group_layout = QHBoxLayout(group)
                
                group_layout.addWidget(QLabel(f"{aug_name}:"))
                
                min_spin = QSpinBox()
                min_spin.setRange(0, 100)
                min_spin.setSuffix("%")
                group_layout.addWidget(QLabel("Min:"))
                group_layout.addWidget(min_spin)
                
                max_spin = QSpinBox()
                max_spin.setRange(0, 100)
                max_spin.setSuffix("%")
                max_spin.setValue(100)
                group_layout.addWidget(QLabel("Max:"))
                group_layout.addWidget(max_spin)
                
                self.intensity_widgets[i] = (min_spin, max_spin)
                self.intensity_layout.addWidget(group)
    
    def get_configuration(self) -> List[AugmentationConfig]:
        configs = []
        for box, intensity in zip(self.selection_boxes, self.intensity_widgets):
            if box.currentIndex() > 0 and intensity is not None:
                min_spin, max_spin = intensity
                aug_type = AugmentationType[box.currentText().replace(' ', '_').upper()]
                configs.append(AugmentationConfig(
                    type=aug_type,
                    min_value=min_spin.value() / 100.0,
                    max_value=max_spin.value() / 100.0
                ))
        return configs
    
    def process_images(self):
        configs = self.get_configuration()
        
        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.summary.clear()
        self.summary.setVisible(True)
        self.status_label.setText("Processing...")
        
        # Create processor and start processing
        self.processor = ImageProcessor(self.input_dir, configs)
        self.processor.progress_signal.connect(self.update_progress)
        self.processor.finished_signal.connect(self.processing_finished)
        self.processor.start()
    
    def update_progress(self, current: int, total: int, message: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
    
    def processing_finished(self, summary: dict):
        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.status_label.setText("Processing complete!")
        
        # Display summary
        summary_text = "Processing Summary:\n\n"
        
        summary_text += "Original Images:\n"
        for subdir, count in summary['original_counts'].items():
            summary_text += f"  {subdir}: {count} images\n"
        
        summary_text += "\nAugmented Images:\n"
        for subdir, count in summary['augmented_counts'].items():
            summary_text += f"  {subdir}: {count} images\n"
        
        summary_text += f"\nTotal Images: {summary['total_images']}\n"
        
        summary_text += "\nAugmentation Parameters:\n"
        for config in self.get_configuration():
            summary_text += (
                f"  {config.type}: "
                f"{config.min_value*100:.0f}% - {config.max_value*100:.0f}%\n"
            )
        
        self.summary.setText(summary_text)