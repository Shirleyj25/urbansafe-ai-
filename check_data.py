import pandas as pd
import os
import matplotlib.pyplot as plt

# Check downloaded files
data_path = "data/raw/"
print("🔍 Checking downloaded files...\n")

csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

if len(csv_files) == 0:
    print("❌ No CSV files found in data/raw/")
else:
    for file in csv_files:
        filepath = os.path.join(data_path, file)
        try:
            df = pd.read_csv(filepath)
            print(f"✅ {file}")
            print(f"   📊 Rows: {len(df):,}")
            print(f"   📋 Columns: {len(df.columns)}")
            print(f"   📏 Size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
            print()
        except Exception as e:
            print(f"❌ Error reading {file}: {str(e)}")

print("🎯 Data check complete!")

# Load datasets for detailed analysis
print("\n📂 Loading collision data for analysis...")
collisions = pd.read_csv("data/raw/dft-road-casualty-statistics-collision-2023.csv", low_memory=False)
print(f"✅ Loaded collisions: {len(collisions):,} records")

# Decode the important categorical values
print("\n🔍 DECODING CATEGORICAL VALUES:")
print("-" * 40)

# Accident Severity (most important for prediction)
print("Accident Severity:")
severity_counts = collisions['accident_severity'].value_counts().sort_index()
severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
for code, count in severity_counts.items():
    severity_name = severity_map.get(code, f'Unknown({code})')
    percentage = (count/len(collisions)*100)
    print(f"  {code} = {severity_name}: {count:,} ({percentage:.1f}%)")

# Weather Conditions
print("\nWeather Conditions (top 5):")
weather_counts = collisions['weather_conditions'].value_counts().head()
weather_map = {1: 'Fine no high winds', 2: 'Raining no high winds', 
               3: 'Snowing no high winds', 4: 'Fine + high winds',
               5: 'Raining + high winds', 8: 'Other', 9: 'Unknown'}
for code, count in weather_counts.items():
    weather_name = weather_map.get(code, f'Code-{code}')
    percentage = (count/len(collisions)*100)
    print(f"  {code} = {weather_name}: {count:,} ({percentage:.1f}%)")

# Road Type
print("\nRoad Type:")
road_counts = collisions['road_type'].value_counts().sort_index()
road_map = {1: 'Roundabout', 2: 'One way street', 3: 'Dual carriageway',
            6: 'Single carriageway', 7: 'Slip road', 9: 'Unknown'}
for code, count in road_counts.items():
    road_name = road_map.get(code, f'Code-{code}')
    percentage = (count/len(collisions)*100)
    print(f"  {code} = {road_name}: {count:,} ({percentage:.1f}%)")

print("\n📈 KEY INSIGHTS:")
print(f"• Total accidents: {len(collisions):,}")
print(f"• Fatal accidents: {len(collisions[collisions['accident_severity']==1]):,}")
print(f"• Most accidents happen in: {weather_map.get(collisions['weather_conditions'].mode()[0], 'Unknown')} weather")
print(f"• Speed limit range: {collisions['speed_limit'].min()}-{collisions['speed_limit'].max()} mph")

# Create visualizations
print("\n📈 CREATING VISUALIZATIONS...")

# 1. Accident Severity Distribution (Pie Chart)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
severity_data = collisions['accident_severity'].value_counts().sort_index()
labels = ['Fatal', 'Serious', 'Slight']
colors = ['red', 'orange', 'lightblue']
plt.pie(severity_data.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Accident Severity Distribution')

# 2. Accidents by Hour of Day
plt.subplot(2, 2, 2)
collisions['hour'] = pd.to_datetime(collisions['time'], format='%H:%M').dt.hour
hourly_accidents = collisions['hour'].value_counts().sort_index()
plt.bar(hourly_accidents.index, hourly_accidents.values, color='skyblue')
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour (24h format)')
plt.ylabel('Number of Accidents')
plt.xticks(range(0, 24, 2))

# 3. Weather vs Accidents
plt.subplot(2, 2, 3)
top_weather = collisions['weather_conditions'].value_counts().head(4)
weather_labels = ['Fine weather', 'Raining', 'Unknown', 'Other']
plt.bar(range(len(top_weather)), top_weather.values, color=['gold', 'lightblue', 'gray', 'orange'])
plt.title('Accidents by Weather Condition')
plt.xlabel('Weather Type')
plt.ylabel('Number of Accidents')
plt.xticks(range(len(top_weather)), weather_labels, rotation=45)

# 4. Road Type vs Accidents
plt.subplot(2, 2, 4)
top_roads = collisions['road_type'].value_counts()
road_labels = ['Single carriageway', 'Dual carriageway', 'Roundabout', 'Unknown', 'One way', 'Slip road']
plt.bar(range(len(top_roads)), top_roads.values, color='lightgreen')
plt.title('Accidents by Road Type')
plt.xlabel('Road Type')
plt.ylabel('Number of Accidents')
plt.xticks(range(len(top_roads)), road_labels, rotation=45)

plt.tight_layout()
plt.savefig('results/accident_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Charts saved to 'results/accident_analysis.png'")

# Weather Impact Analysis
print("\n🌤️ Weather Impact on Accident Severity:")
weather_severity = pd.crosstab(collisions['weather_conditions'], collisions['accident_severity'])
weather_names = {1: 'Fine', 2: 'Raining', 8: 'Other', 9: 'Unknown'}

for weather_code in [1, 2, 8, 9]:
    if weather_code in weather_severity.index:
        total = weather_severity.loc[weather_code].sum()
        fatal_pct = (weather_severity.loc[weather_code, 1] / total * 100) if 1 in weather_severity.columns else 0
        serious_pct = (weather_severity.loc[weather_code, 2] / total * 100) if 2 in weather_severity.columns else 0
        print(f"  {weather_names.get(weather_code, f'Code-{weather_code}')}: {fatal_pct:.1f}% fatal, {serious_pct:.1f}% serious")

# Peak accident hours
peak_hour = hourly_accidents.idxmax()
peak_count = hourly_accidents.max()
print(f"\n⏰ Peak accident hour: {peak_hour}:00 with {peak_count:,} accidents")

print("\n✅ Data analysis complete!")
print("🔍 Next: Check the generated chart in your results/ folder!")

print("\n🤖 BUILDING ACCIDENT RISK PREDICTION MODEL...")
print("="*50)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Prepare data for ML model
print("🔧 Preparing data for machine learning...")

# Select features for prediction
features = ['hour', 'weather_conditions', 'road_type', 'speed_limit', 
           'light_conditions', 'road_surface_conditions', 'day_of_week']

# Create day_of_week feature
collisions['day_of_week'] = pd.to_datetime(collisions['date'], format='%d/%m/%Y').dt.dayofweek

# Prepare feature matrix X and target y
X = collisions[features].copy()
y = collisions['accident_severity'].copy()

# Handle missing values
X = X.fillna(X.mode().iloc[0])

print(f"📊 Features: {features}")
print(f"📊 Training samples: {len(X):,}")
print(f"📊 Target classes: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
print("\n🌲 Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

print("\n📊 Detailed Results:")
target_names = ['Fatal', 'Serious', 'Slight']
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
print("\n🎯 Most Important Features for Prediction:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Example prediction
print("\n🔮 Example Prediction:")
print("Scenario: Rush hour (17:00), Fine weather, Single carriageway, 30mph")
example = [[17, 1, 6, 30, 1, 1, 1]]  # hour, weather, road_type, speed, light, surface, day
prediction = rf_model.predict(example)[0]
probability = rf_model.predict_proba(example)[0]

severity_names = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
print(f"Predicted severity: {severity_names[prediction]}")
print(f"Probabilities: Fatal={probability[0]:.3f}, Serious={probability[1]:.3f}, Slight={probability[2]:.3f}")

print("\n🚀 Machine Learning model complete!")
print("📈 Your UrbanSafe-AI can now predict accident severity!")