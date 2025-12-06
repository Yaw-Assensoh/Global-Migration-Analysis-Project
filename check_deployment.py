import os
import sys

print("🔍 Checking deployment configuration...")

# Check required files
required_files = [
    'requirements.txt',
    '.streamlit/config.toml',
    'streamlit_app.py'
]

all_good = True
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - MISSING")
        all_good = False

# Check pages directory
if os.path.exists('pages'):
    print("✅ pages/ directory")
    pages = [f for f in os.listdir('pages') if f.endswith('.py')]
    print(f"   Found {len(pages)} page(s)")
    for page in pages:
        print(f"   📄 {page}")
else:
    print("❌ pages/ directory - MISSING")
    all_good = False

# Check requirements content
try:
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()
        print(f"✅ requirements.txt has {len(lines)} packages")
        
        # Check for problematic packages
        problem_packages = []
        for line in lines:
            if 'prophet' in line.lower():
                print("   📦 Prophet package found")
            if 'statsmodels' in line.lower():
                print("   📊 Statsmodels package found")
        
except:
    print("❌ Could not read requirements.txt")

print("\n" + "="*50)
if all_good:
    print("🎉 All checks passed! Ready for deployment.")
    print("\nNext steps:")
    print("1. Push to GitHub")
    print("2. Deploy on Streamlit Cloud")
    print("3. Use Python 3.9 in deployment settings")
else:
    print("⚠️  Some issues found. Please fix before deployment.")
    
print("\nTo test locally:")
print("streamlit run streamlit_app.py")
