async function makePredictions(data) {
    const response = await fetch('/reports/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    return await response.json().then(data => data.predictions[0]);
}

async function onFileChange() {
    document.getElementById('benign-attack').classList.add('d-none');
    document.getElementById('dos-attack').classList.add('d-none');
    const file = document.getElementById('file').files[0];
    // Extract the data from the file (.json)
    const data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(JSON.parse(reader.result));
        reader.onerror = error => reject(error);
        reader.readAsText(file);
    });
    // Make predictions
    const prediction = await makePredictions(data);
    if (prediction === 'BENIGN') {
        document.getElementById('benign-attack').classList.remove('d-none');
    } else if (prediction === 'DoS') {
        document.getElementById('dos-attack').classList.remove('d-none');
    }
}
