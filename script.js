let model;
let metadata;

const estado = document.getElementById("estado");
const resultado = document.getElementById("resultado");
const confianzaTexto = document.getElementById("confianza");

const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");


async function cargarModelo() {

    try {

        model = await tf.loadLayersModel("./modelo/model.json");

        const respuesta = await fetch("./modelo/metadata.json");

        metadata = await respuesta.json();

        estado.textContent = "Modelo cargado correctamente";

    } catch (error) {

        estado.textContent = "Error al cargar el modelo";

        console.error("Error:", error);
    }
}


fileInput.addEventListener("change", function () {

    const archivo = fileInput.files[0];

    if (!archivo) {
        return;
    }

    const imageURL = URL.createObjectURL(archivo);

    preview.src = imageURL;

    preview.style.display = "block";

    resultado.textContent = "Analizando imagen...";

    confianzaTexto.textContent = "";

    preview.onload = async function () {

        try {

            const tensor = tf.browser
                .fromPixels(preview)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims(0);

            const predicciones = await model.predict(tensor).data();

            const indice = predicciones.indexOf(
                Math.max(...predicciones)
            );

            const confianza = predicciones[indice] * 100;

            const etiqueta = metadata.labels[indice];

            resultado.textContent =
                "Tipo de moto: " + etiqueta;

            confianzaTexto.textContent =
                "Confianza: " + confianza.toFixed(2) + "%";

            tensor.dispose();

        } catch (error) {

            resultado.textContent =
                "Error al analizar la imagen";

            console.error("Error:", error);
        }
    };
});


cargarModelo();