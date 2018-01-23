# -*- coding: utf-8 -*-
import os
import subprocess
import hashlib
from flask import Flask, request, url_for, send_from_directory, jsonify
from werkzeug import secure_filename
from shutil import copy2
from config import leela_conf

upload_dir = leela_conf.DATA_DIR
download_dir = leela_conf.SAVE_DIR

app = Flask(__name__)

# Max file size  2G
app.config['MAX_CONTENT_LENGTH'] = 2048 * 1024 * 1024
default_job = \
{
    "cmd" : "selfplay",
    "hash" : "65cc94f074951bedeff1428d0aec3df4264a6130be906e18b4e6f63a8536ad4f",
    "options_hash" : "ee21",
    "required_client_version" : "5",
    "leelaz_version" : "0.9",
    "random_seed" : "1",
    "options" : {
        "playouts" : "1000",
        "resignation_percent" : "3",
        "noise" : "true",
        "randomcnt" : "30"
    }
}


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


@app.route('/get-task/7')
def job_info():
    best_net = download_dir + "/best.txt"
    if os.path.exists(best_net):
        global default_job
        sha = sha256_checksum(best_net)
        default_job["hash"] = sha
        sha_file = os.path.join(download_dir, sha)
        if not os.path.exists(sha_file + ".gz"):
            copy2(best_net, sha_file)
            subprocess.call(["gzip", sha_file])
    return jsonify(default_job)


@app.route('/networks/<filename>')
def downloaded_file_by_name(filename):
    return send_from_directory(download_dir, filename, as_attachment=True)


@app.route('/networks')
def downloaded_file():
    iterms = os.listdir(download_dir)
    filename = iterms[0]
    print(filename)
    return send_from_directory(download_dir, filename, as_attachment=True)


@app.route('/submit', methods=['POST'])
def upload_data():
    fs = [request.files.get('sgf'), request.files.get('trainingdata')]

    for f in fs:
        if f:
            filename = secure_filename(f.filename)
            # save
            f.save(os.path.join(upload_dir, filename))
        else:
            return jsonify(False)
    return jsonify(True)


@app.route('/submit-match', methods=['POST'])
def upload_match():
    return jsonify("No need any more!")

if __name__ == '__main__':
    app.run()
