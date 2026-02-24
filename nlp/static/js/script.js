const fileInput = document.getElementById("files");
const fileSlider = document.getElementById("fileSlider");

// Función para manejar el cambio en el input de archivos
fileInput.addEventListener("change", () => {
    const filesArray = Array.from(fileInput.files); // Convertir FileList a Array
    fileSlider.innerHTML = ""; // Limpiar cualquier archivo previamente mostrado

    if (filesArray.length === 0) {
        fileSlider.style.display = "none"; // Ocultar el contenedor si no hay archivos
        return;
    }

    fileSlider.style.display = "block"; // Mostrar el contenedor si hay archivos seleccionados

    // Mostrar los nombres de los archivos seleccionados
    filesArray.forEach((file, index) => {
        const fileElement = document.createElement("div");
        fileElement.className = "file-name";
        fileElement.textContent = `Archivo ${index + 1}: ${file.name}`;
        fileSlider.appendChild(fileElement);
    });
});
