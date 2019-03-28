#!/usr/bin/env bash
# Usage: upload_directory.sh demonstrator_id directory
# Expects an env_id as an environment variable
# Example: env_id='VNCMarioCatcher2-v0' upload_directory.sh demonstrator_EjQjtJoZsLG8dzal /tmp/demo/1475625047-x43pk5scrzsupf-0/

set -eu

if [ "$#" -ne 3 ]; then
    echo "Takes three arguments:"
	echo "    demonstrator_id (e.g. demonstrator_451e8fefb6f54aa8)"
	echo "    directory (e.g. /tmp/demo/1475625047-x43pk5scrzsupf-0/)"
	echo "    s3-bucket (e.g. boxware-vnc-demonstrations-dev)"
	echo "Also, it expects the env_id environment variable to be set, OR"
	echo "    the file /tmp/demo/env_id.txt to contain the env ID"
    echo "Usage example: env_id='flashgames.DuskDrive-v0' ./upload_directory.sh demonstrator_451e8fefb6f54aa8 /tmp/demo/1475625047-x43pk5scrzsupf-0/"
    exit 1
fi

# Argument names
demonstrator_id=$1
directory=$2
bucket=$3

if [[ "$demonstrator_id" != demonstrator_* ]]; then
    echo "Your demonstrator_id must start with the prefix 'demonstrator'. Use the demonstrator_id provided to you by Boxware"
    echo "Usage example: ./upload_directory.sh demonstrator_451e8fefb6f54aa8 /tmp/demo/1475625047-x43pk5scrzsupf-0/"
    exit 1
fi

if ! [ -d "$directory" ]; then
	echo "Error: $directory is not a directory"
	exit 1
fi

echo "Examining $directory"

# Configure AWS with our user: anonymous-public-jiminy-uploader
export AWS_ACCESS_KEY_ID="AKIAJ7V4FOJ3FRK7QFTA"
export AWS_SECRET_ACCESS_KEY="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"

if [ ! -f "$directory/client.fbs" ]; then
	echo "No VNC actions have been recorded into $directory. Not uploading files to S3."
	echo "Connect to localhost:5899 with TurboVNC to record a demonstration"
	exit 1
fi

if [ ! -f "$directory/rewards.demo" ]; then
	echo "No rewards have been copied into $directory. Not uploading files to S3."
	echo "Close your VNC connection before running the upload script"
	exit 1
fi

# if env_id isn't an env variable, maybe we wrote it in demonstration_agent?
if [ -z "${env_id-}" ]; then
	if [ -e '/tmp/demo/env_id.txt' ]; then
		env_id=$(cat '/tmp/demo/env_id.txt')
		echo "Using env_id from file: " "$env_id"
	else
		echo "ERROR: env_id environment variable not set, nor does /tmp/demo/env_id.txt exist. please prefix env_id=ENV_ID_HERE before the command"
		exit 1
	fi

	if [ -z "$env_id" ]; then
		echo "ERROR: env_id still empty... please set the env_id environment variable by prefixing env_id=ENV_ID_HERE before the command"
		exit 1
	fi
fi

echo "Gzipping directory: $directory. This may take a while..."

cd `dirname $directory`  # move to the parent
tarball="$(basename $directory).tar.gz"
tar -zcvf $tarball $(basename $directory)

dest_path="$env_id/$demonstrator_id/$tarball"

# Copy the gzipped dir to S3
echo "Uploading: aws s3 cp $tarball s3://$bucket/$dest_path"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
$DIR/upload.py $tarball $bucket $dest_path
# aws s3 cp $tarball $s3_dest

# Move out of the way so we don't re-upload later
mkdir -p /tmp/uploaded-demos
mv $tarball /tmp/uploaded-demos/

# Don't need the original anymore
rm -rf $directory

echo "Done uploading"
