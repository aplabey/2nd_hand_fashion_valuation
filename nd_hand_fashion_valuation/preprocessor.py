import numpy as np
import pandas as pd
import pickle
import re

#from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import TargetEncoder, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

material_mapping = {
    'Wool': 'Natural Fibers','Cotton': 'Natural Fibers','Silk': 'Natural Fibers','Linen': 'Natural Fibers',
    'Cashmere': 'Natural Fibers','Polyester': 'Synthetic Fibers','Polyamide': 'Synthetic Fibers','Synthetic': 'Synthetic Fibers',
    'Lycra': 'Synthetic Fibers','Spandex': 'Synthetic Fibers','Leather': 'Animal-Based Materials','Suede': 'Animal-Based Materials',
    'Fur': 'Animal-Based Materials','Rabbit': 'Animal-Based Materials','Mink': 'Animal-Based Materials','Fox': 'Animal-Based Materials',
    'Python': 'Animal-Based Materials','Shearling': 'Animal-Based Materials','Alligator': 'Animal-Based Materials',
    'Crocodile': 'Animal-Based Materials','Chinchilla': 'Animal-Based Materials','Pony-style calfskin': 'Animal-Based Materials',
    'Water snake': 'Animal-Based Materials','Eel': 'Animal-Based Materials','Gold': 'Metals','Platinum': 'Metals','Titanium': 'Metals',
    'Silver': 'Metals','Steel': 'Metals','Gold plated': 'Metals','White gold': 'Metals','Yellow gold': 'Metals',
    'Silver Plated': 'Metals','Silver Gilt': 'Metals','Cotton - elasthane': 'Blends','Denim - Jeans': 'Blends','Wicker': 'Blends',
    'Vegan leather': 'Other Materials','Velvet': 'Other Materials','Lace': 'Other Materials','Glitter': 'Other Materials',
    'Tweed': 'Other Materials','Vinyl': 'Other Materials','Exotic leathers': 'Other Materials','Plastic': 'Other Materials',
    'Patent leather': 'Other Materials','Astrakhan': 'Other Materials','Ostrich': 'Other Materials','Sponge': 'Other Materials',
    'Rubber': 'Other Materials','Wood': 'Other Materials','Ceramic': 'Other Materials','Glass': 'Other Materials',
    'Pearl': 'Other Materials','Chain': 'Metals','Pearls': 'Other Materials','Varan': 'Other Materials','Not specified': 'Unspecified'
}

shipping_days_mapping = {
    '1-2 days': 1.5,
    '3-5 days': 4,
    '6-7 days': 6.5,
    'More than 7 days': 8  # or another appropriate value based on your data
}

color_mapping = {
    'Grey': 'Neutrals','Navy': 'Neutrals','White': 'Neutrals','Black': 'Neutrals','Beige': 'Neutrals','Ecru': 'Neutrals','Anthracite': 'Neutrals','Charcoal': 'Neutrals',
    'Khaki': 'Neutrals','Camel': 'Neutrals','camel': 'Neutrals','Brown': 'Neutrals','White / Black': 'Neutrals','Beige / Grey': 'Neutrals','brown/black': 'Neutrals',

    'Red': 'Colorful','Green': 'Colorful','Blue': 'Colorful','Turquoise': 'Colorful','Yellow': 'Colorful','Pink': 'Colorful',
    'Orange': 'Colorful','Burgundy': 'Colorful','Purple': 'Colorful','Bordeaux': 'Colorful',

    'Metallic': 'Special','Gold': 'Special','Silver': 'Special','silver/black': 'Special','Multicolour': 'Special'
}

def preprocess_text(text):
    if pd.isna(text):
        return ""  # Handle missing values
    text = str(text)  # Ensure input is a string
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def preprocess_features(df):

    df = df.drop_duplicates()

    df = df[df['price_usd'] <= 10000]
    df['usually_ships_within'] = df['usually_ships_within'].replace(np.nan, "1-2 days")
    df['has_cross_border_fees'] = df['has_cross_border_fees'].replace(np.nan, False)
    df['buyers_fees'] = df['buyers_fees'].replace(np.nan, 0)
    df = df.dropna()

    df['material_group'] = df['product_material'].map(material_mapping)
    df['shipping_days'] = df['usually_ships_within'].map(shipping_days_mapping)
    df['color_group'] = df['product_color'].map(color_mapping)

    df['gender_binary'] = df['product_gender_target'].map({'Men': 0, 'Women': 1})
    # Apply text preprocessing
    df['cleaned_description'] = df['product_description'].apply(preprocess_text)
    df = select_features(df)

    return df

def select_features(df):
    df = df.drop(['product_like_count',	'buyers_fees','product_gender_target','product_id', 'product_type',
            'brand_url', 'brand_id',
            'product_material', 'product_color', 'product_name',
            'product_description', 'product_keywords', 'warehouse_name', 'seller_id',
            'seller_pass_rate', 'seller_num_followers', 'seller_country', 'seller_price',
            'seller_earning', 'seller_community_rank', 'seller_username',
            'seller_num_products_listed', 'sold', 'reserved', 'available',
            'in_stock', 'should_be_gone', 'has_cross_border_fees', 'usually_ships_within'], axis=1)


    return df


def preproc_pipe(X, y_log):

    # Define the transformers for numerical and categorical features
    num_transformer = make_pipeline(RobustScaler())

    # Categorical feature transformers
    cat_transformer = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

    # Ordinal feature transformers
    ord_enc_product_condition = OrdinalEncoder(categories=[['Fair condition', 'Good condition', 'Very good condition', 'Never worn', 'Never worn, with tag']])
    ord_enc_seller_badge = OrdinalEncoder(categories=[['Common', 'Trusted', 'Expert']])

    # Transform brands
    brand_transformer = TargetEncoder(categories='auto', target_type='continuous', cv=5)

    # Define the TF-IDF vectorizer for text data
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # Limit to 1000 features

    preproc = ColumnTransformer(
        [
            ('num', num_transformer, ['seller_products_sold', 'shipping_days']),
            ('cat', cat_transformer, ['product_category', 'product_season', 'material_group', 'color_group']),
            ('ord_condition', ord_enc_product_condition, ['product_condition']),
            ('ord_badge', ord_enc_seller_badge, ['seller_badge']),
            ('brand_enc', brand_transformer, ['brand_name']),
            ('tfidf',tfidf_vectorizer,'cleaned_description')


        ],
        remainder='passthrough'
    )


    X_processed = preproc.fit_transform(X, y_log)

    # generate a pickle file of this pipeline as it will then be used for transforming the X_pred in api call
    with open("models/pipeline.pkl", "wb") as file:
            pickle.dump(preproc, file)
            print("----- pipeline pickle has been generated -----")

    return X_processed


def api_preprocessor(df):

    df['material_group'] = df['material_group'].map(material_mapping)
    df['shipping_days'] = df['shipping_days'].map(shipping_days_mapping)
    df['color_group'] = df['color_group'].map(color_mapping)

    df['gender_binary'] = df['gender_binary'].map({'Men': 0, 'Women': 1})
    # Apply text preprocessing
    df['cleaned_description'] = df['cleaned_description'].apply(preprocess_text)

    return df
