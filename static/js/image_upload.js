let dropzone = document.querySelector('.dropzone');
let fileInput = document.querySelector('#fileInput');
let previewImage = document.querySelector('#previewImage');
let deleteIcon = document.querySelector('#deleteIcon');
let submitBtn = document.querySelector('#submitBtn');
let file;

function handleFileSelect(event) {
  let file = event.target.files[0];
  if (file.type === 'image/png' || file.type === 'image/jpeg' || file.type === 'image/jpg') {
    let reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.style.display = 'block';
      deleteIcon.style.display = 'inline-block';
      dropzone.classList.remove('dragover');
      dropzone.classList.add('disabled');
      fileInput.classList.add('disabled');
      submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  } else {
    alert('Please choose an image file. Image Extention Should be in jpg or jpeg or png');
  }
}

fileInput.addEventListener('change', handleFileSelect);

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  file = e.dataTransfer.files[0]; // remove the "let" keyword
  if (file.type === 'image/png' || file.type === 'image/jpeg' || file.type === 'image/jpg') {
    let reader = new FileReader();



    let formData = new FormData();
    formData.append('image', file);

    fileInput.files = e.dataTransfer.files;

    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.style.display = 'block';
      deleteIcon.style.display = 'inline-block';
      dropzone.classList.remove('dragover');
      dropzone.classList.add('disabled');
      fileInput.classList.add('disabled');
      submitBtn.disabled = false;
    }
    reader.readAsDataURL(file);
  } else {
    alert('Please choose an image file. Image Extention Should be in jpg or jpeg or png');
  }
});

deleteIcon.addEventListener('click', () => {
  previewImage.style.display = 'none';
  deleteIcon.style.display = 'none';
  fileInput.value = '';
  submitBtn.disabled = true;
  dropzone.classList.remove('disabled');
  fileInput.classList.remove('disabled');
});
