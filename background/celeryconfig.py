"""
Celery configuration for QuickBooks Data Cleaner.
"""

# Broker settings
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
task_acks_late = True
worker_max_tasks_per_child = 1000

# Task routing
task_routes = {
    'tasks.clean_file': {'queue': 'cleaning'}
}

# Result settings
result_expires = 3600  # 1 hour 