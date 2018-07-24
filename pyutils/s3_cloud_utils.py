# Utils for transparent access to S3 or local files
#
# Copyright (C) 2018  Author: Misha Orel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import io
import boto3
from botocore.exceptions import ClientError
import dill
import skimage.io
import os
import sys


S3Prefix = "s3://"

class S3Object:
    def __init__(self, profileName):
        self.s3_ = boto3.Session(profile_name=profileName).resource('s3')

    def getS3Object(self, s3Path):
        return self.s3_.Bucket(s3Path.split('/')[2]).Object('/'.join(s3Path.split('/')[3:]))

    @staticmethod
    def normalizeS3Path(path):
        path = path.split('://')[-1]
        while path and path[0] == '/':
            path = path[1:]
        return S3Prefix + path

    def copyFileLocally(self, path, local_path):
        fin = self.binaryReadIo(path)
        with open(local_path, 'wb') as fout:
            fout.write(fin.read())

    def binaryReadIo(self, path):
        """Read a binary file from either S3 or a local path."""
        if path.startswith(S3Prefix):
            try:
                obj = self.getS3Object(path)
                return io.BytesIO(obj.get()['Body'].read())
            except ClientError as ex:
                if ex.response['Error']['Code'] == 'NoSuchKey':
                    print('S3 error for %s' % path)
                    return None
                else:
                    raise ex
        else:
            return open(path, 'rb')
    
    def textReadIo(self, path):
        """Read a text file from either S3 or a local path."""
        if path.startswith(S3Prefix):
            try:
                obj = self.getS3Object(path)
                return io.StringIO(obj.get()['Body'].read().decode('utf-8'))
            except ClientError as ex:
                if ex.response['Error']['Code'] == 'NoSuchKey':
                    print('S3 error for %s' % path)
                    return None
                else:
                    raise ex
        else:
            return open(path, 'r')

    def pickleObject(self, path):
        bio = self.binaryReadIo(path)
        return dill.load(bio) if bio is not None else None

    def imageRead(self, path):
        bio = self.binaryReadIo(path)
        return skimage.io.imread(bio) if bio is not None else None

