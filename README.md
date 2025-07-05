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

### Step 3: Install and Start Redis
Celery requires a message broker to handle task queuing. In this case, we are using Redis. You need to install and start Redis.

#### Install Redis:
**For Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install redis-server
```

#### Start Redis:
Ensure Redis is running:

```bash
sudo systemctl start redis
sudo systemctl enable redis  # Optional, to start Redis on boot
```

#### Check Redis Status:
You can check whether Redis is running with:

```bash
sudo systemctl status redis
```

### Step 4: Configure Celery and Start the Worker
Celery handles background tasks, such as model training. Once Redis is running, you need to start the Celery worker.

In a separate terminal window, activate the virtual environment and run the Celery worker for the `automl` app:

```bash
celery -A automlapp worker --loglevel=info
```

### Step 5: Make Migrations and Migrate the Database
Create the initial migrations and apply them to set up the database.

```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 6: Create a Superuser (Optional)
If you want to access the Django admin panel, create a superuser.

```bash
python manage.py createsuperuser
```

### Step 7: Run the Development Server
Start the Django development server to run the app.

```bash
python manage.py runserver
```

### Step 8: Access the App
Open your web browser and visit the following URL to access the app:

```
http://127.0.0.1:8000/
```

### Step 9: Run Tests
To ensure everything is working as expected, run the tests using `pytest`.

```bash
pytest
```

This will verify that your functions and app are functioning properly.
