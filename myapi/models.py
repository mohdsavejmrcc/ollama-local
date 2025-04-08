from django.db import models
from django.utils import timezone
import uuid

# Create your models here.
class Chunk(models.Model):
    chunk_id=models.AutoField(primary_key=True)
    file_name=models.CharField(max_length=255)
    file_path=models.CharField(max_length=512)
    chunk_text=models.TextField()
    page_number=models.IntegerField(null=True,blank=True)

    embedding=models.JSONField(null=True,blank=True)
    metadata=models.JSONField(null=True,blank=True)
    created_at=models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Chunk {self.chunk_id} from {self.file_name}"
    class Meta:
        db_table='chunks'
        indexes=[
            models.Index(fields=['file_name'])
        ]