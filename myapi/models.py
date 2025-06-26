from django.db import models
from django.utils import timezone
import uuid
from django.contrib.auth.models import User
from mongoengine import (
    Document,
    EmbeddedDocument,
    StringField,
    BooleanField,
    DateTimeField,
    EmbeddedDocumentListField,
    ReferenceField
)
from datetime import datetime
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

class Chat(models.Model):
    name = models.CharField(max_length=255)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
class Message(models.Model):
    chat = models.ForeignKey(Chat, related_name='messages', on_delete=models.CASCADE)
    request = models.BooleanField()  # True = user, False = bot
    response = models.BooleanField()
    visible =  models.BooleanField()  # True = bot, False = user
    message = models.TextField()
    prompt = models.TextField(null=True, blank=True)  # Only for user messages
    timestamp = models.DateTimeField(auto_now_add=True)