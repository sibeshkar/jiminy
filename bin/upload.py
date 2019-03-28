#!/usr/bin/env python

import os
import sys
import threading

import boto3
import s3transfer

# Public anonymous uploader credentials. These are not secret
AWS_ACCESS_KEY_ID = "AKIAJ7V4FOJ3FRK7QFTA"
AWS_SECRET_ACCESS_KEY = "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"


class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            print("%s / %s  (%.2f%%)" % (self._seen_so_far,
                                             self._size, percentage), end='\r')
            sys.stdout.flush()


def upload(src, bucket_name, dest_path):
    print('Starting upload (using uploader v2)...')
    client = boto3.client(
        service_name='s3',
        region_name='us-west-2',
        # verify=False,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )


    config = s3transfer.TransferConfig(
        max_concurrency=10,
        num_download_attempts=10,
    )

    transfer = s3transfer.S3Transfer(client, config)
    transfer.upload_file(src, bucket_name, dest_path,
                     callback=ProgressPercentage(src))

if __name__ == '__main__':
    src = sys.argv[1]
    bucket_name = sys.argv[2]
    dest_path = sys.argv[3]

    upload(src, bucket_name, dest_path)
