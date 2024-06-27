from bottle import route, run, template, request, static_file, url, get, post, response, error, abort, redirect, os
static_path = 'D:\projects\spot-the-difference-generate\static\\'
@route('/machigai/<filename>')
def home(filename):
    return static_file(filename, root=static_path)

@route('/process', method='POST')
def process():
    data = request.files.getall('up')
    for upload in data:
        name, ext = os.path.splitext(upload.filename)
        if ext not in ('.png', '.jpg', '.jpeg'):
            return 'File extension not allowed.'

    for upload in data:
        print(upload.filename)
        save_path = upload.filename
        upload.save(save_path)
    return 'OK'


if __name__ == "__main__":
    run(host='192.168.11.22', port=80, debug=True, reloader=True)