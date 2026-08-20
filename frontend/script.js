// =====================================================
// AI CROP DOCTOR
// =====================================================

const API_URL =
    "http://127.0.0.1:5000/predict";


// Elements
const form =
    document.getElementById(
        "predictionForm"
    );

const imageInput =
    document.getElementById(
        "imageInput"
    );

const preview =
    document.getElementById(
        "preview"
    );

const previewContainer =
    document.getElementById(
        "previewContainer"
    );

const removeImage =
    document.getElementById(
        "removeImage"
    );

const uploadBox =
    document.getElementById(
        "uploadBox"
    );

const result =
    document.getElementById(
        "result"
    );

const predictButton =
    document.getElementById(
        "predictButton"
    );


// =====================================================
// IMAGE SELECT
// =====================================================

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];

        if (!file) {
            return;
        }

        if (
            !file.type.startsWith(
                "image/"
            )
        ) {

            alert(
                "Please select an image."
            );

            imageInput.value = "";

            return;
        }


        // Preview
        const imageURL =
            URL.createObjectURL(
                file
            );

        preview.src =
            imageURL;

        previewContainer.classList.add(
            "show"
        );


        // Update upload box
        uploadBox.style.display =
            "none";


        // Clear previous result
        result.innerHTML = "";
    }
);


// =====================================================
// REMOVE IMAGE
// =====================================================

removeImage.addEventListener(
    "click",
    function () {

        imageInput.value = "";

        preview.src = "";

        previewContainer.classList.remove(
            "show"
        );

        uploadBox.style.display =
            "flex";

        result.innerHTML = "";
    }
);


// =====================================================
// FORM SUBMIT
// =====================================================

form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        const file =
            imageInput.files[0];


        if (!file) {

            result.innerHTML = `
                <div class="error">
                    Please upload a leaf image first.
                </div>
            `;

            return;
        }


        // Loading
        result.innerHTML = `
            <div class="loading">
                🌿 AI is analyzing your leaf...
                <br><br>
                Please wait.
            </div>
        `;


        predictButton.disabled =
            true;

        predictButton.innerHTML =
            "⏳ Analyzing...";


        // FormData
        const formData =
            new FormData();

        formData.append(
            "image",
            file
        );


        try {

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok ||
                data.status === "error"
            ) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );
            }


            // Format disease
            const disease =
                data.disease
                    .replaceAll(
                        "_",
                        " "
                    )
                    .replaceAll(
                        "___",
                        " "
                    );


            const confidence =
                Number(
                    data.confidence
                ).toFixed(2);


            // Result
            result.innerHTML = `

                <div class="prediction-result">

                    <h2>
                        🌱 Prediction Result
                    </h2>

                    <p>
                        <strong>
                            Disease:
                        </strong>
                        ${disease}
                    </p>

                    <p>
                        <strong>
                            Confidence:
                        </strong>
                        ${confidence}%
                    </p>

                </div>

            `;

        }


        catch (error) {

            console.error(
                error
            );


            result.innerHTML = `

                <div class="error">

                    <strong>
                        ❌ Prediction Failed
                    </strong>

                    <br><br>

                    ${error.message}

                    <br><br>

                    Make sure the Flask
                    backend is running
                    on port 5000.

                </div>

            `;
        }


        finally {

            predictButton.disabled =
                false;

            predictButton.innerHTML =
                `
                <span>🔍</span>
                Analyze Leaf
                <span class="arrow">→</span>
                `;
        }

    }
);