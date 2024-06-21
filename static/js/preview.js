function previewFile(hoge){
    var fileData = new FileReader();
    fileData.onload = (function() {
      document.getElementById('preview').src = fileData.result;
    });
    fileData.readAsDataURL(hoge.files[0]);
    console.log('preview is done!')
  }