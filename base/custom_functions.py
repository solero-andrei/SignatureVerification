import os

from django.core.files.storage import FileSystemStorage
from django.conf import settings

class Helper():

    def SaveOSignature(self, file_name, file):
        fs = FileSystemStorage(location=directory_to_save)
        fs.save(file_name, file)

    def __Create_Signature_Directory(self, id, firstname, middlename, lastname):
        dir_name = f'{id}-{firstname}_{middlename}_{lastname}'
        signature_dir = os.path.join(settings.MEDIA_ROOT, dir_name)
        if os.path.isdir(signature_dir) == False:
            os.makedirs(signature_dir)