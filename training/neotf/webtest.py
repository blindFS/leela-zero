# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory, jsonify
from werkzeug import secure_filename

upload_dir = "./data"
download_dir = "./save"

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
        "playouts" : "10",
        "resignation_percent" : "3",
        "noise" : "true",
        "randomcnt" : "30"
    }
}


def allowed_file(filename):
    return True

@app.route('/get-task/7')
def job_info():
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
        if f and allowed_file(f.filename):
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
