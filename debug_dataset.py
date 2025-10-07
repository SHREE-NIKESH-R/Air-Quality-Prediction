import pandas as pd
import os

print("="*70)
print("DATASET DEBUGGER")
print("="*70)

# Find CSV files
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"\nCSV files found: {csv_files}")

if len(csv_files) == 0:
    print("\n❌ No CSV files found!")
    exit()

# Let user choose
if len(csv_files) == 1:
    dataset_file = csv_files[0]
else:
    print("\nSelect file:")
    for i, f in enumerate(csv_files):
        print(f"{i+1}. {f}")
    choice = int(input("Enter number: ")) - 1
    dataset_file = csv_files[choice]

print(f"\n{'='*70}")
print(f"Analyzing: {dataset_file}")
print(f"{'='*70}")

# Load dataset
df = pd.read_csv(dataset_file)

# Basic info
print(f"\n📊 BASIC INFO:")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print(f"   File size: {os.path.getsize(dataset_file) / (1024*1024):.2f} MB")

# Show all columns
print(f"\n📋 ALL COLUMNS ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    nulls = df[col].isnull().sum()
    null_pct = (nulls / len(df)) * 100
    print(f"   {i:2}. {col:40} | Type: {str(dtype):10} | Nulls: {nulls:6} ({null_pct:5.1f}%)")

# Show first few rows
print(f"\n📄 FIRST 5 ROWS:")
print(df.head())

# Show data types
print(f"\n🔍 DATA TYPES:")
print(df.dtypes)

# Show sample values for each column
print(f"\n💡 SAMPLE VALUES (first non-null value from each column):")
for col in df.columns:
    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "ALL NULL"
    print(f"   {col:40} : {sample}")

# Check for numeric columns
print(f"\n🔢 NUMERIC COLUMNS:")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    for col in numeric_cols:
        print(f"   ✅ {col}")
else:
    print("   ❌ No numeric columns found!")

# Statistical summary
if numeric_cols:
    print(f"\n📈 STATISTICS FOR NUMERIC COLUMNS:")
    print(df[numeric_cols].describe())

# Look for pollutant-like columns
print(f"\n🔍 SEARCHING FOR AIR QUALITY COLUMNS:")
keywords = {
    'PM2.5': ['pm2.5', 'pm25', 'pm_2.5', 'pm 2.5'],
    'PM10': ['pm10', 'pm_10', 'pm 10'],
    'NO2': ['no2', 'no_2', 'no 2'],
    'CO': ['co', 'carbon'],
    'SO2': ['so2', 'so_2', 'so 2'],
    'O3': ['o3', 'o_3', 'ozone'],
    'AQI': ['aqi', 'air quality', 'index']
}

found_matches = {}
for standard_name, search_terms in keywords.items():
    matches = []
    for col in df.columns:
        col_lower = col.lower().strip()
        for term in search_terms:
            if term in col_lower:
                matches.append(col)
                break
    if matches:
        found_matches[standard_name] = matches
        print(f"   ✅ {standard_name:8} → {matches}")
    else:
        print(f"   ❌ {standard_name:8} → Not found")

# Check if any matches found
if not found_matches:
    print(f"\n⚠️  WARNING: No air quality columns detected!")
    print(f"\n💡 TIP: Your dataset columns should include:")
    print(f"   - PM2.5 or PM25")
    print(f"   - PM10")
    print(f"   - NO2")
    print(f"   - CO")
    print(f"   - SO2")
    print(f"   - O3")
    print(f"   - AQI (optional)")
    print(f"\n📝 Please check if you downloaded the correct dataset!")

# Check for object (string) columns that should be numeric
print(f"\n⚠️  STRING COLUMNS THAT MIGHT NEED CONVERSION:")
object_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    # Try to check if it looks numeric
    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
    if sample and any(c.isdigit() for c in str(sample)):
        print(f"   ⚠️  {col:40} : {sample} (might be numeric)")

# Final recommendations
print(f"\n{'='*70}")
print("📋 RECOMMENDATIONS:")
print("="*70)

if not found_matches:
    print("\n❌ PROBLEM: No pollutant columns found!")
    print("\n✅ SOLUTION:")
    print("   1. Check if you downloaded the correct dataset")
    print("   2. Dataset should have columns like: PM2.5, PM10, NO2, CO, SO2, O3")
    print("   3. Try a different dataset from Kaggle")
    print("\n📌 RECOMMENDED DATASETS:")
    print("   - Global Air Quality Dataset")
    print("   - Air Quality Data in India (city_day.csv)")
    print("   - Beijing Air Quality Data")
elif len(found_matches) < 4:
    print(f"\n⚠️  ISSUE: Only {len(found_matches)} pollutants found")
    print("   Need at least 4 pollutants for good predictions")
    print(f"\n   Found: {', '.join(found_matches.keys())}")
    print(f"   Missing: {', '.join(set(keywords.keys()) - set(found_matches.keys()))}")
else:
    print(f"\n✅ GOOD: Found {len(found_matches)} air quality parameters!")
    print(f"   Detected: {', '.join(found_matches.keys())}")
    print("\n   The dataset looks usable!")
    
    # Check data quality
    total_nulls = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    null_percentage = (total_nulls / total_cells) * 100
    
    print(f"\n📊 DATA QUALITY:")
    print(f"   Total records: {len(df):,}")
    print(f"   Null values: {total_nulls:,} ({null_percentage:.1f}%)")
    
    if null_percentage > 50:
        print(f"   ⚠️  High percentage of missing data!")
    elif null_percentage > 20:
        print(f"   ⚠️  Moderate missing data")
    else:
        print(f"   ✅ Good data quality")

print(f"\n{'='*70}")
print("Run this script to understand your dataset structure!")
print("Then we can fix the model_training script accordingly.")
print("="*70)