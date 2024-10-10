
### Step 1: Set Up a Virtual Environment
First, create and activate a virtual environment to isolate your project dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Required Packages
Install the necessary dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Make Migrations and Migrate the Database
Create the initial migrations and apply them to set up the database.

```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 4: Create a Superuser (Optional)
If you want to access the Django admin panel, create a superuser.

```bash
python manage.py createsuperuser
```

### Step 5: Run the Development Server
Start the Django development server to run the app.

```bash
python manage.py runserver
```

### Step 6: Access the App
Open your web browser and visit the following URL to access the app:

```
http://127.0.0.1:8000/
```

### Step 7: Run Tests
To ensure everything is working as expected, run the tests using `pytest`.

```bash
pytest
```

This will verify that your functions and app are functioning properly.