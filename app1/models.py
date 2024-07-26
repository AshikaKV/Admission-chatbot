from django.db import models

class UserQuery(models.Model):
    query_text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return self.query_text
