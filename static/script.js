function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("predictionResult").innerHTML = `<h3>Predicted Disease: ${data.prediction}</h3>`;
    })
    .catch(error => console.error("Error:", error));
}
