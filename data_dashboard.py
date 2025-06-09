import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataAnalyticsDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.original_df = None
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        self.setWindowTitle("Advanced Data Analytics Dashboard")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #000000; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; color: #777777; }

            QPushButton { 
                background-color: #00ffcc; 
                border: none; 
                color: #000000; 
                padding: 8px 16px; 
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1affd5; }
            QPushButton:pressed { background-color: #00ccaa; }

            QComboBox, QCheckBox { 
                padding: 5px; 
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #00ffff;
            }

            QTabWidget::pane { border: 1px solid #333; }

            QTabBar::tab { 
                background-color: #2a2a2a; 
                padding: 8px 12px; 
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #00ffff;
            }
            QTabBar::tab:selected { 
                background-color: #00ffcc; 
                color: #000000; 
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("📊 Advanced Data Analytics Dashboard")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50; margin: 10px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # File operations
        self.load_btn = QPushButton("📁 Load Data File")
        self.load_btn.clicked.connect(self.load_data)
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        
        header_layout.addWidget(self.load_btn)
        header_layout.addWidget(self.export_btn)
        main_layout.addLayout(header_layout)
        
        # Status bar
        self.status_label = QLabel("Ready to load data...")
        self.status_label.setStyleSheet("padding: 5px; background-color: #e8f4fd; border-radius: 4px;")
        main_layout.addWidget(self.status_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Initialize tabs
        self.init_data_tab()
        self.init_visualization_tab()
        self.init_analytics_tab()
        self.init_ml_tab()
        
    def init_data_tab(self):
        # Data Overview Tab
        data_widget = QWidget()
        layout = QVBoxLayout(data_widget)
        
        # Data info panel
        info_layout = QHBoxLayout()
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setStyleSheet("font-weight: bold; color: #34495e;")
        info_layout.addWidget(self.data_info_label)
        info_layout.addStretch()
        
        # Data filtering controls
        filter_group = QGroupBox("Data Filtering & Modification")
        filter_layout = QGridLayout(filter_group)
        
        # Column selection
        filter_layout.addWidget(QLabel("Select Columns:"), 0, 0)
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.column_list.itemSelectionChanged.connect(self.filter_columns)
        filter_layout.addWidget(self.column_list, 1, 0, 3, 1)
        
        # Row filtering
        filter_layout.addWidget(QLabel("Filter Rows:"), 0, 1)
        self.filter_column_combo = QComboBox()
        self.filter_operator_combo = QComboBox()
        self.filter_operator_combo.addItems(['>', '<', '>=', '<=', '==', '!=', 'contains'])
        self.filter_value_input = QLineEdit()
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.apply_row_filter)
        self.reset_filter_btn = QPushButton("Reset Data")
        self.reset_filter_btn.clicked.connect(self.reset_data)
        
        filter_layout.addWidget(self.filter_column_combo, 1, 1)
        filter_layout.addWidget(self.filter_operator_combo, 2, 1)
        filter_layout.addWidget(self.filter_value_input, 3, 1)
        filter_layout.addWidget(self.apply_filter_btn, 4, 1)
        filter_layout.addWidget(self.reset_filter_btn, 5, 1)
        
        layout.addLayout(info_layout)
        layout.addWidget(filter_group)
        
        # Data table
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        
        self.tab_widget.addTab(data_widget, "📋 Data Overview")
        
    def init_visualization_tab(self):
        # Visualization Tab
        viz_widget = QWidget()
        layout = QHBoxLayout(viz_widget)
        
        # Control panel
        control_panel = QGroupBox("Visualization Controls")
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)
        
        # Chart type selection
        control_layout.addWidget(QLabel("Chart Type:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            'Histogram', 'Scatter Plot', 'Line Plot', 'Bar Chart', 
            'Box Plot', 'Heatmap', 'Correlation Matrix', 'Pair Plot'
        ])
        self.chart_type_combo.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(self.chart_type_combo)
        
        # Axis selection
        control_layout.addWidget(QLabel("X-Axis:"))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(self.x_axis_combo)
        
        control_layout.addWidget(QLabel("Y-Axis:"))
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(self.y_axis_combo)
        
        # Color grouping
        control_layout.addWidget(QLabel("Color By:"))
        self.color_combo = QComboBox()
        self.color_combo.addItem("None")
        self.color_combo.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(self.color_combo)
        
        # Size selection
        control_layout.addWidget(QLabel("Size By:"))
        self.size_combo = QComboBox()
        self.size_combo.addItem("None")
        self.size_combo.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(self.size_combo)
        
        # Show/hide options
        self.show_trend_cb = QCheckBox("Show Trend Line")
        self.show_trend_cb.stateChanged.connect(self.update_visualization)
        control_layout.addWidget(self.show_trend_cb)
        
        self.log_scale_cb = QCheckBox("Log Scale")
        self.log_scale_cb.stateChanged.connect(self.update_visualization)
        control_layout.addWidget(self.log_scale_cb)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
        # Visualization area
        self.viz_canvas = MatplotlibCanvas()
        layout.addWidget(self.viz_canvas)
        
        self.tab_widget.addTab(viz_widget, "📈 Visualizations")
        
    def init_analytics_tab(self):
        # Analytics Tab
        analytics_widget = QWidget()
        layout = QVBoxLayout(analytics_widget)
        
        # Analytics controls
        controls_layout = QHBoxLayout()
        
        self.analytics_btn = QPushButton("🔍 Generate Analytics Report")
        self.analytics_btn.clicked.connect(self.generate_analytics)
        controls_layout.addWidget(self.analytics_btn)
        
        self.correlation_btn = QPushButton("📊 Correlation Analysis")
        self.correlation_btn.clicked.connect(self.correlation_analysis)
        controls_layout.addWidget(self.correlation_btn)
        
        self.outliers_btn = QPushButton("⚠️ Detect Outliers")
        self.outliers_btn.clicked.connect(self.detect_outliers)
        controls_layout.addWidget(self.outliers_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Analytics results
        self.analytics_text = QTextEdit()
        self.analytics_text.setStyleSheet("background-color: white; border: 1px solid #ddd;")
        layout.addWidget(self.analytics_text)
        
        self.tab_widget.addTab(analytics_widget, "🔍 Analytics")
        
    def init_ml_tab(self):
        # Machine Learning Tab
        ml_widget = QWidget()
        layout = QHBoxLayout(ml_widget)
        
        # ML Control panel
        ml_control_panel = QGroupBox("Machine Learning Models")
        ml_control_layout = QVBoxLayout(ml_control_panel)
        ml_control_panel.setMaximumWidth(350)
        
        # Target variable selection
        ml_control_layout.addWidget(QLabel("Target Variable:"))
        self.target_combo = QComboBox()
        ml_control_layout.addWidget(self.target_combo)
        
        # Model type selection
        ml_control_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            'Auto Classification', 'Auto Regression', 'K-Means Clustering', 
            'PCA Analysis', 'Random Forest', 'Feature Importance'
        ])
        ml_control_layout.addWidget(self.model_type_combo)
        
        # Feature selection
        ml_control_layout.addWidget(QLabel("Select Features:"))
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        ml_control_layout.addWidget(self.feature_list)
        
        # ML parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Test size
        params_layout.addWidget(QLabel("Test Size:"))
        self.test_size_slider = QSlider(Qt.Horizontal)
        self.test_size_slider.setRange(10, 50)
        self.test_size_slider.setValue(20)
        self.test_size_label = QLabel("20%")
        self.test_size_slider.valueChanged.connect(
            lambda v: self.test_size_label.setText(f"{v}%")
        )
        params_layout.addWidget(self.test_size_slider)
        params_layout.addWidget(self.test_size_label)
        
        # Number of clusters (for K-means)
        params_layout.addWidget(QLabel("Number of Clusters:"))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(3)
        params_layout.addWidget(self.n_clusters_spin)
        
        ml_control_layout.addWidget(params_group)
        
        # Train model button
        self.train_model_btn = QPushButton("🚀 Train Model")
        self.train_model_btn.clicked.connect(self.train_ml_model)
        ml_control_layout.addWidget(self.train_model_btn)
        
        # Export model button
        self.export_model_btn = QPushButton("📤 Export Model")
        self.export_model_btn.clicked.connect(self.export_model)
        self.export_model_btn.setEnabled(False)
        ml_control_layout.addWidget(self.export_model_btn)
        
        ml_control_layout.addStretch()
        layout.addWidget(ml_control_panel)
        
        # ML Results area
        ml_results_widget = QWidget()
        ml_results_layout = QVBoxLayout(ml_results_widget)
        
        # Results text
        self.ml_results_text = QTextEdit()
        self.ml_results_text.setStyleSheet("background-color: white; border: 1px solid #fff;")
        ml_results_layout.addWidget(self.ml_results_text)
        
        # ML visualization
        self.ml_canvas = MatplotlibCanvas()
        ml_results_layout.addWidget(self.ml_canvas)
        
        layout.addWidget(ml_results_widget)
        
        self.tab_widget.addTab(ml_widget, "🤖 Machine Learning")
    
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Data File", "", 
            "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                if file_path.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(file_path)
                else:
                    self.df = pd.read_csv(file_path)
                
                self.original_df = self.df.copy()
                self.populate_ui_elements()
                self.update_data_table()
                self.status_label.setText(f"✅ Loaded: {file_path} ({len(self.df)} rows, {len(self.df.columns)} columns)")
                self.export_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def populate_ui_elements(self):
        if self.df is None:
            return
            
        columns = list(self.df.columns)
        numeric_columns = list(self.df.select_dtypes(include=[np.number]).columns)
        
        # Clear existing items
        self.column_list.clear()
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        self.color_combo.clear()
        self.size_combo.clear()
        self.filter_column_combo.clear()
        self.target_combo.clear()
        self.feature_list.clear()
        
        # Populate column list
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.column_list.addItem(item)
        
        # Populate combo boxes
        self.x_axis_combo.addItems(columns)
        self.y_axis_combo.addItems(numeric_columns)
        self.color_combo.addItem("None")
        self.color_combo.addItems(columns)
        self.size_combo.addItem("None")
        self.size_combo.addItems(numeric_columns)
        self.filter_column_combo.addItems(columns)
        self.target_combo.addItems(columns)
        
        # Populate feature list
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.feature_list.addItem(item)
        
        # Update data info
        info_text = f"Shape: {self.df.shape} | "
        info_text += f"Numeric: {len(numeric_columns)} | "
        info_text += f"Categorical: {len(columns) - len(numeric_columns)} | "
        info_text += f"Missing: {self.df.isnull().sum().sum()}"
        self.data_info_label.setText(info_text)
    
    def update_data_table(self):
        if self.df is None:
            return
            
        self.data_table.setRowCount(min(1000, len(self.df)))  # Limit to 1000 rows for performance
        self.data_table.setColumnCount(len(self.df.columns))
        self.data_table.setHorizontalHeaderLabels(self.df.columns)
        
        for i in range(min(1000, len(self.df))):
            for j, col in enumerate(self.df.columns):
                item = QTableWidgetItem(str(self.df.iloc[i, j]))
                self.data_table.setItem(i, j, item)
        
        self.data_table.resizeColumnsToContents()
    
    def filter_columns(self):
        if self.df is None:
            return
            
        selected_columns = []
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_columns.append(item.text())
        
        if selected_columns:
            self.df = self.original_df[selected_columns].copy()
            self.update_data_table()
            self.status_label.setText(f"✅ Filtered to {len(selected_columns)} columns")
    
    def apply_row_filter(self):
        if self.df is None:
            return
            
        column = self.filter_column_combo.currentText()
        operator = self.filter_operator_combo.currentText()
        value = self.filter_value_input.text()
        
        if not value:
            return
        
        try:
            if operator == 'contains':
                mask = self.df[column].astype(str).str.contains(value, case=False, na=False)
            else:
                # Try to convert value to numeric if possible
                try:
                    numeric_value = float(value)
                    if operator == '>':
                        mask = self.df[column] > numeric_value
                    elif operator == '<':
                        mask = self.df[column] < numeric_value
                    elif operator == '>=':
                        mask = self.df[column] >= numeric_value
                    elif operator == '<=':
                        mask = self.df[column] <= numeric_value
                    elif operator == '==':
                        mask = self.df[column] == numeric_value
                    elif operator == '!=':
                        mask = self.df[column] != numeric_value
                except ValueError:
                    # String comparison
                    if operator == '==':
                        mask = self.df[column].astype(str) == value
                    elif operator == '!=':
                        mask = self.df[column].astype(str) != value
                    else:
                        QMessageBox.warning(self, "Warning", "Invalid operator for string values")
                        return
            
            self.df = self.df[mask].copy()
            self.update_data_table()
            self.status_label.setText(f"✅ Filtered to {len(self.df)} rows")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Filter error: {str(e)}")
    
    def reset_data(self):
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.update_data_table()
            self.status_label.setText("✅ Data reset to original")
    
    def update_visualization(self):
        if self.df is None:
            return
        
        chart_type = self.chart_type_combo.currentText()
        x_col = self.x_axis_combo.currentText()
        y_col = self.y_axis_combo.currentText()
        color_col = self.color_combo.currentText() if self.color_combo.currentText() != "None" else None
        size_col = self.size_combo.currentText() if self.size_combo.currentText() != "None" else None
        
        self.viz_canvas.plot_data(self.df, chart_type, x_col, y_col, color_col, size_col,
                                 self.show_trend_cb.isChecked(), self.log_scale_cb.isChecked())
    
    def generate_analytics(self):
        if self.df is None:
            return
        
        analytics_report = "📊 COMPREHENSIVE DATA ANALYTICS REPORT\n"
        analytics_report += "=" * 50 + "\n\n"
        
        # Basic statistics
        analytics_report += "📈 BASIC STATISTICS:\n"
        analytics_report += f"Dataset Shape: {self.df.shape}\n"
        analytics_report += f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
        analytics_report += f"Missing Values: {self.df.isnull().sum().sum()}\n"
        analytics_report += f"Duplicate Rows: {self.df.duplicated().sum()}\n\n"
        
        # Numeric columns analysis
        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            analytics_report += "🔢 NUMERIC COLUMNS ANALYSIS:\n"
            analytics_report += str(numeric_df.describe()) + "\n\n"
            
            # Skewness and kurtosis
            analytics_report += "📐 DISTRIBUTION METRICS:\n"
            for col in numeric_df.columns:
                skew = numeric_df[col].skew()
                kurt = numeric_df[col].kurtosis()
                analytics_report += f"{col}: Skewness={skew:.3f}, Kurtosis={kurt:.3f}\n"
            analytics_report += "\n"
        
        # Categorical columns analysis
        categorical_df = self.df.select_dtypes(include=['object'])
        if not categorical_df.empty:
            analytics_report += "🏷️ CATEGORICAL COLUMNS ANALYSIS:\n"
            for col in categorical_df.columns:
                unique_vals = categorical_df[col].nunique()
                most_common = categorical_df[col].mode().iloc[0] if not categorical_df[col].mode().empty else "None"
                analytics_report += f"{col}: {unique_vals} unique values, Most common: {most_common}\n"
            analytics_report += "\n"
        
        # Data quality assessment
        analytics_report += "✅ DATA QUALITY ASSESSMENT:\n"
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            if missing_pct > 0:
                analytics_report += f"{col}: {missing_pct:.1f}% missing values\n"
        
        self.analytics_text.setText(analytics_report)
    
    def correlation_analysis(self):
        if self.df is None:
            return
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 numeric columns for correlation analysis")
            return
        
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append(
                        (correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val)
                    )
        
        report = "🔗 CORRELATION ANALYSIS REPORT\n"
        report += "=" * 40 + "\n\n"
        
        if strong_correlations:
            report += "💪 STRONG CORRELATIONS (|r| > 0.7):\n"
            for col1, col2, corr in strong_correlations:
                direction = "positive" if corr > 0 else "negative"
                report += f"{col1} ↔ {col2}: {corr:.3f} ({direction})\n"
        else:
            report += "No strong correlations found (|r| > 0.7)\n"
        
        report += "\n📊 FULL CORRELATION MATRIX:\n"
        report += str(correlation_matrix.round(3))
        
        self.analytics_text.setText(report)
    
    def detect_outliers(self):
        if self.df is None:
            return
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            QMessageBox.warning(self, "Warning", "No numeric columns found for outlier detection")
            return
        
        outlier_report = "⚠️ OUTLIER DETECTION REPORT\n"
        outlier_report += "=" * 35 + "\n\n"
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(numeric_df)) * 100
            
            outlier_report += f"📊 {col}:\n"
            outlier_report += f"  Outliers: {outlier_count} ({outlier_percentage:.1f}%)\n"
            outlier_report += f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]\n"
            if outlier_count > 0 and outlier_count <= 10:
                outlier_report += f"  Values: {outliers.tolist()}\n"
            outlier_report += "\n"
        
        self.analytics_text.setText(outlier_report)
    
    def train_ml_model(self):
        if self.df is None:
            return
        
        model_type = self.model_type_combo.currentText()
        target_col = self.target_combo.currentText()
        
        # Get selected features
        selected_features = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked and item.text() != target_col:
                selected_features.append(item.text())
        
        if not selected_features:
            QMessageBox.warning(self, "Warning", "Please select at least one feature")
            return
        
        try:
            if model_type in ['Auto Classification', 'Auto Regression', 'Random Forest']:
                self.train_supervised_model(model_type, target_col, selected_features)
            elif model_type == 'K-Means Clustering':
                self.train_clustering_model(selected_features)
            elif model_type == 'PCA Analysis':
                self.perform_pca_analysis(selected_features)
            elif model_type == 'Feature Importance':
                self.analyze_feature_importance(target_col, selected_features)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model training failed: {str(e)}")
    
    def train_supervised_model(self, model_type, target_col, features):
        # Prepare data
        X = self.df[features].copy()
        y = self.df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else y.mode().iloc[0])
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Determine if classification or regression
        is_classification = (
            model_type == 'Auto Classification' or 
            (model_type in ['Auto Regression', 'Random Forest'] and 
             (pd.api.types.is_object_dtype(y) or y.nunique() < 20))
        )
        
        if is_classification and pd.api.types.is_object_dtype(y):
            if 'target_encoder' not in self.label_encoders:
                self.label_encoders['target_encoder'] = LabelEncoder()
            y = self.label_encoders['target_encoder'].fit_transform(y.astype(str))
        
        # Split data
        test_size = self.test_size_slider.value() / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            metric_name = "Accuracy"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = mean_squared_error(y_test, y_pred)  # RMSE
            metric_name = "RMSE"
        
        # Store model
        self.ml_models[model_type] = {
            'model': model,
            'features': features,
            'target': target_col,
            'is_classification': is_classification,
            'score': score,
            'metric': metric_name
        }
        
        # Generate results
        results = f"🤖 {model_type.upper()} RESULTS\n"
        results += "=" * 40 + "\n\n"
        results += f"Target Variable: {target_col}\n"
        results += f"Features Used: {', '.join(features)}\n"
        results += f"Training Samples: {len(X_train)}\n"
        results += f"Test Samples: {len(X_test)}\n"
        results += f"{metric_name}: {score:.4f}\n\n"
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            results += "📊 FEATURE IMPORTANCE:\n"
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for _, row in importance_df.head(10).iterrows():
                results += f"{row['Feature']}: {row['Importance']:.4f}\n"
        
        self.ml_results_text.setText(results)
        
        # Visualize results
        self.visualize_ml_results(model_type, y_test, y_pred, importance_df if hasattr(model, 'feature_importances_') else None)
        self.export_model_btn.setEnabled(True)
    
    def train_clustering_model(self, features):
        # Prepare data
        X = self.df[features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = X[col].fillna(X[col].mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-means
        n_clusters = self.n_clusters_spin.value()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        # Store model
        self.ml_models['K-Means'] = {
            'model': kmeans,
            'features': features,
            'clusters': clusters,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters
        }
        
        # Add clusters to dataframe
        self.df['Cluster'] = clusters
        
        # Generate results
        results = f"🎯 K-MEANS CLUSTERING RESULTS\n"
        results += "=" * 35 + "\n\n"
        results += f"Features Used: {', '.join(features)}\n"
        results += f"Number of Clusters: {n_clusters}\n"
        results += f"Silhouette Score: {silhouette_avg:.4f}\n\n"
        
        # Cluster statistics
        results += "📊 CLUSTER STATISTICS:\n"
        cluster_stats = self.df.groupby('Cluster')[features].mean()
        results += str(cluster_stats.round(3)) + "\n\n"
        
        # Cluster sizes
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        results += "📈 CLUSTER SIZES:\n"
        for cluster, count in cluster_counts.items():
            percentage = (count / len(clusters)) * 100
            results += f"Cluster {cluster}: {count} samples ({percentage:.1f}%)\n"
        
        self.ml_results_text.setText(results)
        
        # Visualize clustering results
        self.visualize_clustering_results(X_scaled, clusters, features)
        self.export_model_btn.setEnabled(True)
    
    def perform_pca_analysis(self, features):
        # Prepare data
        X = self.df[features].copy()
        
        # Handle missing values and encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = X[col].fillna(X[col].mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Store model
        self.ml_models['PCA'] = {
            'model': pca,
            'features': features,
            'transformed_data': X_pca
        }
        
        # Generate results
        results = f"🔍 PCA ANALYSIS RESULTS\n"
        results += "=" * 30 + "\n\n"
        results += f"Original Features: {len(features)}\n"
        results += f"Principal Components: {len(pca.components_)}\n\n"
        
        # Explained variance
        results += "📊 EXPLAINED VARIANCE:\n"
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        for i, (var, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
            results += f"PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)\n"
            if i >= 9:  # Show first 10 components
                break
        
        # Find components that explain 95% of variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        results += f"\nComponents for 95% variance: {n_components_95}\n"
        
        # Top feature loadings for first few components
        results += "\n🎯 TOP FEATURE LOADINGS:\n"
        for i in range(min(3, len(pca.components_))):
            results += f"\nPC{i+1}:\n"
            loadings = pd.DataFrame({
                'Feature': features,
                'Loading': np.abs(pca.components_[i])
            }).sort_values('Loading', ascending=False)
            
            for _, row in loadings.head(5).iterrows():
                results += f"  {row['Feature']}: {row['Loading']:.4f}\n"
        
        self.ml_results_text.setText(results)
        
        # Visualize PCA results
        self.visualize_pca_results(pca, X_pca, features)
        self.export_model_btn.setEnabled(True)
    
    def analyze_feature_importance(self, target_col, features):
        # Use Random Forest for feature importance
        X = self.df[features].copy()
        y = self.df[target_col].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = X[col].fillna(X[col].mean())
        
        # Handle target variable
        if y.dtype == 'object':
            if 'target_encoder' not in self.label_encoders:
                self.label_encoders['target_encoder'] = LabelEncoder()
            y = self.label_encoders['target_encoder'].fit_transform(y.astype(str))
            is_classification = True
        else:
            y = y.fillna(y.mean())
            is_classification = y.nunique() < 20
        
        # Train model for feature importance
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        # Calculate feature importance
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Store results
        self.ml_models['Feature Importance'] = {
            'model': model,
            'importance': importance_df,
            'features': features,
            'target': target_col
        }
        
        # Generate results
        results = f"🎯 FEATURE IMPORTANCE ANALYSIS\n"
        results += "=" * 35 + "\n\n"
        results += f"Target Variable: {target_col}\n"
        results += f"Analysis Type: {'Classification' if is_classification else 'Regression'}\n\n"
        
        results += "📊 FEATURE IMPORTANCE RANKING:\n"
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            percentage = (row['Importance'] / importance_df['Importance'].sum()) * 100
            results += f"{i:2d}. {row['Feature']}: {row['Importance']:.4f} ({percentage:.1f}%)\n"
        
        # Identify most important features
        top_features = importance_df.head(5)['Feature'].tolist()
        results += f"\n🌟 TOP 5 FEATURES: {', '.join(top_features)}\n"
        
        # Low importance features (potential candidates for removal)
        low_importance = importance_df[importance_df['Importance'] < 0.01]
        if not low_importance.empty:
            results += f"\n⚠️ LOW IMPORTANCE FEATURES ({len(low_importance)}):\n"
            results += ', '.join(low_importance['Feature'].tolist())
        
        self.ml_results_text.setText(results)
        
        # Visualize feature importance
        self.visualize_feature_importance(importance_df)
        self.export_model_btn.setEnabled(True)
    
    def visualize_ml_results(self, model_type, y_test, y_pred, importance_df=None):
        self.ml_canvas.figure.clear()
        
        if model_type in ['Auto Classification', 'Auto Regression', 'Random Forest']:
            if importance_df is not None:
                # Create subplots
                ax1 = self.ml_canvas.figure.add_subplot(221)
                ax2 = self.ml_canvas.figure.add_subplot(222)
                ax3 = self.ml_canvas.figure.add_subplot(223)
                
                # Actual vs Predicted
                ax1.scatter(y_test, y_pred, alpha=0.6)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax1.set_xlabel('Actual Values')
                ax1.set_ylabel('Predicted Values')
                ax1.set_title('Actual vs Predicted')
                
                # Residuals plot
                residuals = y_test - y_pred
                ax2.scatter(y_pred, residuals, alpha=0.6)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residuals Plot')
                
                # Feature importance
                top_features = importance_df.head(10)
                ax3.barh(range(len(top_features)), top_features['Importance'])
                ax3.set_yticks(range(len(top_features)))
                ax3.set_yticklabels(top_features['Feature'])
                ax3.set_xlabel('Importance')
                ax3.set_title('Feature Importance')
        
        self.ml_canvas.figure.tight_layout()
        self.ml_canvas.draw()
    
    def visualize_clustering_results(self, X_scaled, clusters, features):
        self.ml_canvas.figure.clear()
        
        if X_scaled.shape[1] >= 2:
            # If more than 2 features, use PCA for visualization
            if X_scaled.shape[1] > 2:
                pca_viz = PCA(n_components=2)
                X_viz = pca_viz.fit_transform(X_scaled)
                xlabel, ylabel = 'First Principal Component', 'Second Principal Component'
            else:
                X_viz = X_scaled
                xlabel, ylabel = features[0], features[1]
            
            ax = self.ml_canvas.figure.add_subplot(111)
            scatter = ax.scatter(X_viz[:, 0], X_viz[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('Clustering Results')
            self.ml_canvas.figure.colorbar(scatter)
        
        self.ml_canvas.figure.tight_layout()
        self.ml_canvas.draw()
    
    def visualize_pca_results(self, pca, X_pca, features):
        self.ml_canvas.figure.clear()
        
        # Create subplots
        ax1 = self.ml_canvas.figure.add_subplot(221)
        ax2 = self.ml_canvas.figure.add_subplot(222)
        ax3 = self.ml_canvas.figure.add_subplot(223)
        
        # Explained variance
        ax1.bar(range(1, min(11, len(pca.explained_variance_ratio_) + 1)), 
                pca.explained_variance_ratio_[:10])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        
        # Cumulative explained variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumvar) + 1), cumvar, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        
        # PCA scatter plot (first two components)
        if X_pca.shape[1] >= 2:
            ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
            ax3.set_xlabel('First Principal Component')
            ax3.set_ylabel('Second Principal Component')
            ax3.set_title('PCA Scatter Plot')
        
        self.ml_canvas.figure.tight_layout()
        self.ml_canvas.draw()
    
    def visualize_feature_importance(self, importance_df):
        self.ml_canvas.figure.clear()
        
        ax = self.ml_canvas.figure.add_subplot(111)
        top_features = importance_df.head(15)
        ax.barh(range(len(top_features)), top_features['Importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance Analysis')
        
        self.ml_canvas.figure.tight_layout()
        self.ml_canvas.draw()
    
    def export_model(self):
        if not self.ml_models:
            QMessageBox.warning(self, "Warning", "No trained models to export")
            return
        
        # Let user choose which model to export
        models = list(self.ml_models.keys())
        model_name, ok = QInputDialog.getItem(self, "Export Model", "Select model to export:", models, 0, False)
        
        if ok and model_name:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Model", f"{model_name.lower().replace(' ', '_')}_model.pkl", 
                "Pickle files (*.pkl);;All files (*.*)"
            )
            
            if file_path:
                try:
                    import pickle
                    model_data = {
                        'model': self.ml_models[model_name]['model'],
                        'scaler': self.scaler,
                        'label_encoders': self.label_encoders,
                        'features': self.ml_models[model_name].get('features', []),
                        'model_type': model_name,
                        'metadata': {
                            'target': self.ml_models[model_name].get('target', 'Unknown'),
                            'score': self.ml_models[model_name].get('score', 0),
                            'metric': self.ml_models[model_name].get('metric', 'Unknown')
                        }
                    }
                    
                    with open(file_path, 'wb') as f:
                        pickle.dump(model_data, f)
                    
                    QMessageBox.information(self, "Success", f"Model exported successfully to {file_path}")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to export model: {str(e)}")
    
    def export_results(self):
        if self.df is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "analysis_results.xlsx", 
            "Excel files (*.xlsx);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path) as writer:
                        self.df.to_excel(writer, sheet_name='Processed_Data', index=False)
                        
                        # Add analytics summary
                        summary_data = {
                            'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Duplicate Rows'],
                            'Value': [len(self.df), len(self.df.columns), 
                                     self.df.isnull().sum().sum(), self.df.duplicated().sum()]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Add correlation matrix if available
                        numeric_df = self.df.select_dtypes(include=[np.number])
                        if not numeric_df.empty:
                            numeric_df.corr().to_excel(writer, sheet_name='Correlations')
                else:
                    self.df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Results exported successfully to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")


class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(12, 8))
        super().__init__(self.figure)
        self.setParent(None)
        
    def plot_data(self, df, chart_type, x_col, y_col, color_col=None, size_col=None, 
                  show_trend=False, log_scale=False):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        try:
            if chart_type == 'Histogram':
                if x_col and x_col in df.columns:
                    if df[x_col].dtype in ['object', 'category']:
                        df[x_col].value_counts().plot(kind='bar', ax=ax)
                        ax.set_title(f'Distribution of {x_col}')
                    else:
                        ax.hist(df[x_col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Histogram of {x_col}')
            
            elif chart_type == 'Scatter Plot' and x_col and y_col:
                if x_col in df.columns and y_col in df.columns:
                    scatter_data = df[[x_col, y_col]].dropna()
                    if not scatter_data.empty:
                        if color_col and color_col in df.columns:
                            if df[color_col].dtype in ['object', 'category']:
                                for category in df[color_col].unique():
                                    mask = df[color_col] == category
                                    ax.scatter(df[mask][x_col], df[mask][y_col], 
                                             label=str(category), alpha=0.6)
                                ax.legend()
                            else:
                                scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], 
                                                   cmap='viridis', alpha=0.6)
                                self.figure.colorbar(scatter)
                        else:
                            ax.scatter(df[x_col], df[y_col], alpha=0.6)
                        
                        if show_trend:
                            z = np.polyfit(scatter_data[x_col], scatter_data[y_col], 1)
                            p = np.poly1d(z)
                            ax.plot(scatter_data[x_col], p(scatter_data[x_col]), "r--", alpha=0.8)
                        
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f'{x_col} vs {y_col}')
            
            elif chart_type == 'Line Plot' and x_col and y_col:
                if x_col in df.columns and y_col in df.columns:
                    line_data = df[[x_col, y_col]].dropna().sort_values(x_col)
                    ax.plot(line_data[x_col], line_data[y_col], marker='o', alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'{y_col} over {x_col}')
            
            elif chart_type == 'Bar Chart' and x_col:
                if x_col in df.columns:
                    if df[x_col].dtype in ['object', 'category']:
                        value_counts = df[x_col].value_counts().head(20)
                        value_counts.plot(kind='bar', ax=ax)
                        ax.set_title(f'Distribution of {x_col}')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel('Count')
                    else:
                        # For numeric data, create bins
                        bins = pd.cut(df[x_col].dropna(), bins=10)
                        bins.value_counts().sort_index().plot(kind='bar', ax=ax)
                        ax.set_title(f'Distribution of {x_col}')
            
            elif chart_type == 'Box Plot' and y_col:
                if y_col in df.columns:
                    if color_col and color_col in df.columns:
                        df.boxplot(column=y_col, by=color_col, ax=ax)
                        ax.set_title(f'Box Plot of {y_col} by {color_col}')
                    else:
                        ax.boxplot(df[y_col].dropna())
                        ax.set_ylabel(y_col)
                        ax.set_title(f'Box Plot of {y_col}')
            
            elif chart_type == 'Heatmap':
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    correlation_matrix = numeric_df.corr()
                    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                    ax.set_xticks(range(len(correlation_matrix.columns)))
                    ax.set_yticks(range(len(correlation_matrix.columns)))
                    ax.set_xticklabels(correlation_matrix.columns, rotation=45)
                    ax.set_yticklabels(correlation_matrix.columns)
                    ax.set_title('Correlation Heatmap')
                    self.figure.colorbar(im)
            
            elif chart_type == 'Correlation Matrix':
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', 
                               center=0, ax=ax, fmt='.2f')
                    ax.set_title('Correlation Matrix')
            
            elif chart_type == 'Pair Plot':
                numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :5]  # Limit to 5 columns
                if len(numeric_df.columns) >= 2:
                    pd.plotting.scatter_matrix(numeric_df, ax=ax, alpha=0.6, figsize=(10, 10))
                    ax.set_title('Pair Plot')
            
            if log_scale and chart_type in ['Scatter Plot', 'Line Plot']:
                try:
                    ax.set_yscale('log')
                except:
                    pass
                    
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
        
        self.figure.tight_layout()
        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application icon and properties
    app.setApplicationName("Data Analytics Dashboard")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Analytics Pro")
    
    window = DataAnalyticsDashboard()
    window.show()
    
    sys.exit(app.exec_())
