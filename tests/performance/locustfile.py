# tests/performance/locustfile.py
from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def get_health(self):
        self.client.get("/health")

    @task
    def get_centers(self):
        self.client.get("/centers")

    @task
    def generate_forecast(self):
        self.client.get("/forecast?center=KASARA&item=CHILAPI&days=7")
