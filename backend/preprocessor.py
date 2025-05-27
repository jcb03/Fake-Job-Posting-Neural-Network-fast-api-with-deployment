import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

class ImbalanceAwarePreprocessor:
    """Advanced preprocessor with class imbalance handling capabilities"""
    
    def __init__(self, handle_imbalance='smote'):
        """
        Initialize preprocessor with imbalance handling options
        
        Parameters:
        handle_imbalance: 'smote', 'oversample', 'undersample', 'smote_tomek', 'class_weights', None
        """
        # Removed print statement for deployment
        self.handle_imbalance = handle_imbalance
        
        # Text vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased for better representation
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore rare terms
            max_df=0.95,  # Ignore too common terms
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        # Other preprocessors
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.class_weights = None
        
        # Initialize sampling methods
        if handle_imbalance == 'smote':
            self.sampler = SMOTE(random_state=42, k_neighbors=3)
        elif handle_imbalance == 'oversample':
            self.sampler = RandomOverSampler(random_state=42)
        elif handle_imbalance == 'undersample':
            self.sampler = RandomUnderSampler(random_state=42)
        elif handle_imbalance == 'smote_tomek':
            self.sampler = SMOTETomek(random_state=42)
        else:
            self.sampler = None
    
    def clean_text(self, text):
        """Enhanced text cleaning for job postings"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def combine_text_features(self, df):
        """Combine all text features with weighted importance"""
        text_columns = ['title', 'description', 'company_profile', 'requirements', 'benefits']
        existing_text_cols = [col for col in text_columns if col in df.columns]
        
        combined_text = []
        for idx, row in df.iterrows():
            text_parts = []
            
            # Title gets higher weight (repeated 3 times)
            if 'title' in existing_text_cols:
                title_clean = self.clean_text(row['title'])
                if title_clean:
                    text_parts.extend([title_clean] * 3)
            
            # Description gets medium weight (repeated 2 times)
            if 'description' in existing_text_cols:
                desc_clean = self.clean_text(row['description'])
                if desc_clean:
                    text_parts.extend([desc_clean] * 2)
            
            # Other fields get normal weight
            for col in existing_text_cols:
                if col not in ['title', 'description']:
                    cleaned_text = self.clean_text(row[col])
                    if cleaned_text:
                        text_parts.append(cleaned_text)
            
            combined_text.append(' '.join(text_parts))
        
        return combined_text
    
    def encode_categorical_features(self, df):
        """Encode categorical features with handling for high cardinality"""
        possible_categorical = [
            'employment_type', 'required_experience', 'required_education',
            'industry', 'function', 'location', 'department'
        ]
        
        categorical_columns = [col for col in possible_categorical if col in df.columns]
        
        encoded_features = []
        
        for col in categorical_columns:
            df_copy = df.copy()
            df_copy[col] = df_copy[col].fillna('Unknown')
            
            # Handle high cardinality columns
            if col in ['location', 'industry', 'function', 'department']:
                # Keep only top 30 categories, group rest as 'Other'
                top_categories = df_copy[col].value_counts().head(30).index
                df_copy[col] = df_copy[col].apply(lambda x: x if x in top_categories else 'Other')
            
            # Label encode with unseen category handling
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df_copy[col])
            else:
                # Handle unseen categories during prediction - ADDED THIS
                try:
                    encoded = self.label_encoders[col].transform(df_copy[col])
                except ValueError:
                    # Handle unseen categories by mapping them to 'Unknown'
                    known_classes = set(self.label_encoders[col].classes_)
                    df_copy[col] = df_copy[col].apply(lambda x: x if x in known_classes else 'Unknown')
                    
                    # If 'Unknown' is not in known classes, add it
                    if 'Unknown' not in known_classes:
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                    
                    encoded = self.label_encoders[col].transform(df_copy[col])
            
            encoded_features.append(encoded.reshape(-1, 1))
        
        if encoded_features:
            return np.hstack(encoded_features)
        else:
            return np.array([]).reshape(len(df), 0)
    
    def extract_numerical_features(self, df):
        """Extract and engineer numerical features"""
        numerical_features = []
        
        # Binary features
        binary_columns = ['has_company_logo', 'telecommuting', 'has_questions']
        existing_binary = [col for col in binary_columns if col in df.columns]
        
        for col in existing_binary:
            values = df[col].fillna(0).astype(int).values.reshape(-1, 1)
            numerical_features.append(values)
        
        # Text-derived features
        if 'description' in df.columns:
            # Description metrics
            desc_length = df['description'].fillna('').astype(str).apply(len).values.reshape(-1, 1)
            word_count = df['description'].fillna('').astype(str).apply(lambda x: len(x.split())).values.reshape(-1, 1)
            exclamation_count = df['description'].fillna('').astype(str).apply(lambda x: x.count('!')).values.reshape(-1, 1)
            caps_ratio = df['description'].fillna('').astype(str).apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
            ).values.reshape(-1, 1)
            
            numerical_features.extend([desc_length, word_count, exclamation_count, caps_ratio])
        
        if 'title' in df.columns:
            # Title metrics
            title_length = df['title'].fillna('').astype(str).apply(len).values.reshape(-1, 1)
            title_word_count = df['title'].fillna('').astype(str).apply(lambda x: len(x.split())).values.reshape(-1, 1)
            title_caps_ratio = df['title'].fillna('').astype(str).apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
            ).values.reshape(-1, 1)
            
            numerical_features.extend([title_length, title_word_count, title_caps_ratio])
        
        # Additional engineered features
        if 'requirements' in df.columns:
            has_requirements = (df['requirements'].fillna('').astype(str).apply(len) > 0).astype(int).values.reshape(-1, 1)
            req_length = df['requirements'].fillna('').astype(str).apply(len).values.reshape(-1, 1)
            numerical_features.extend([has_requirements, req_length])
        
        if 'salary_range' in df.columns:
            has_salary = (~df['salary_range'].isna()).astype(int).values.reshape(-1, 1)
            numerical_features.append(has_salary)
        
        if 'company_profile' in df.columns:
            has_company_profile = (df['company_profile'].fillna('').astype(str).apply(len) > 0).astype(int).values.reshape(-1, 1)
            numerical_features.append(has_company_profile)
        
        if numerical_features:
            return np.hstack(numerical_features)
        else:
            return np.array([]).reshape(len(df), 0)
    
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced data"""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, class_weights))
        
        return self.class_weights
    
    def fit_transform(self, df, y=None):
        """Fit preprocessor and transform data with imbalance handling"""
        if y is None:
            y = df['fraudulent'].values
        
        # 1. Process text features
        combined_text = self.combine_text_features(df)
        text_features = self.tfidf_vectorizer.fit_transform(combined_text).toarray()
        
        # 2. Process categorical features
        categorical_features = self.encode_categorical_features(df)
        
        # 3. Process numerical features
        numerical_features = self.extract_numerical_features(df)
        
        # 4. Combine all features
        all_features = [text_features]
        
        if categorical_features.shape[1] > 0:
            all_features.append(categorical_features)
        
        if numerical_features.shape[1] > 0:
            all_features.append(numerical_features)
        
        final_features = np.hstack(all_features)
        
        # 5. Scale features
        final_features = self.scaler.fit_transform(final_features)
        
        # 6. Handle class imbalance
        if self.handle_imbalance == 'class_weights':
            self.calculate_class_weights(y)
            X_balanced, y_balanced = final_features, y
        elif self.sampler is not None:
            X_balanced, y_balanced = self.sampler.fit_resample(final_features, y)
        else:
            X_balanced, y_balanced = final_features, y
        
        return X_balanced, y_balanced
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        # ADDED: Handle single row predictions from API
        if len(df) == 1:
            # Ensure all expected columns exist with default values
            expected_columns = [
                'title', 'description', 'company_profile', 'requirements', 'benefits',
                'location', 'employment_type', 'required_experience', 'required_education',
                'industry', 'function', 'fraudulent'
            ]
            
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = '' if col != 'fraudulent' else 0
        
        combined_text = self.combine_text_features(df)
        text_features = self.tfidf_vectorizer.transform(combined_text).toarray()
        
        categorical_features = self.encode_categorical_features(df)
        numerical_features = self.extract_numerical_features(df)
        
        all_features = [text_features]
        if categorical_features.shape[1] > 0:
            all_features.append(categorical_features)
        if numerical_features.shape[1] > 0:
            all_features.append(numerical_features)
        
        final_features = np.hstack(all_features)
        final_features = self.scaler.transform(final_features)
        
        return final_features
