from bottle import route, run, static_file
 
@route('/file/<filename:path>')
def static(filename):
    return static_file(filename, root="D:\projects\spot-the-difference-generate\static\\")
 
if __name__ == "__main__":
    run(host='localhost', port=80, debug=True, reloader=True)