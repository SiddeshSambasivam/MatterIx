### **Environment setup**
Install the necessary dependecies in a seperate virtual environment
```bash
# Create a virtual environment during development to avoid dependency issues
pip install -r requirements.txt

# Before submitting a PR, run the unittests locally
pytest -v
```

### **Contributing**

1. Fork it

2. Create your feature branch
  ```
  git checkout -b feature/new_feature
  ```
  
3. Commit your changes
  ```bash
  git commit -m 'add new feature'
  ```
  
4. Push to the branch
  ```bash
  git push origin feature/new_feature
  ```
  
5. Create a new pull request (PR)
