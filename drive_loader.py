import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

fid = '1B9YuUXWXvLgibfHPBGDeRDIROq0hsWDZ'

def drive_auth():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive

def create_folder(drive, folder_name, parent_folder_id):  
    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{"kind": "drive#fileLink", "id": parent_folder_id}]
    }
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    return folder['id']

def save_to_drive(drive, ckpt_folder_id, colab_file_path):
    uploaded  = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id":ckpt_folder_id}]})
    uploaded.SetContentFile(colab_file_path)
    uploaded.Upload()
    print('Uploaded file with ID {}'.format(uploaded.get('id')))

def load_from_drive(drive, colab_download_path, ckpt_number, general_ckpt_folder_id):
    folders_list = drive.ListFile({'q': "\'"+general_ckpt_folder_id+"\' in parents"}).GetList()
    folders_list = [folder for folder in folders_list if not folder['labels']['trashed']]
    folder_name = 'ckpt'+ckpt_number
    for folder in folders_list:
        if folder['title'] == folder_name:
            folder_id = folder['id']
    file_list = drive.ListFile({'q': "\'"+folder_id+"\' in parents"}).GetList()
    for f in file_list:
        print('title: %s, id: %s' % (f['title'], f['id']))
        fname = os.path.join(colab_download_path, f['title'])
        print('downloading to {}'.format(fname))
        f_ = drive.CreateFile({'id': f['id']})
        f_.GetContentFile(fname)