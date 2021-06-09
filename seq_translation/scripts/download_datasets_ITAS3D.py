import os
from download_gdrive import *

file_id = 'GI965WdyFAPMUqfbESFUeIki43kUAqUP'
chpt_path = './datasets/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
destination = os.path.join(chpt_path, 's001_sample.zip')
download_file_from_google_drive(file_id, destination) 
unzip_file(destination, chpt_path)