# Generated by Django 4.1.7 on 2023-03-30 10:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="signaturedata",
            name="SignaturesID",
        ),
        migrations.CreateModel(
            name="Signatures",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("Signature", models.ImageField(blank=True, null=True, upload_to="")),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="base.signaturedata",
                    ),
                ),
            ],
        ),
    ]