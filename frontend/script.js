// =========================================================
// AI CROP DOCTOR
// FRONTEND JAVASCRIPT
// CNN + GEMINI
// =========================================================


// =========================================================
// API
// =========================================================

const API_URL = "http://127.0.0.1:5000/predict";


// =========================================================
// ELEMENTS
// =========================================================

const form =
    document.getElementById("predictionForm");

const imageInput =
    document.getElementById("imageInput");

const preview =
    document.getElementById("preview");

const previewContainer =
    document.getElementById("previewContainer");

const removeImage =
    document.getElementById("removeImage");

const uploadBox =
    document.getElementById("uploadBox");

const result =
    document.getElementById("result");

const predictButton =
    document.getElementById("predictButton");


// =========================================================
// CHECK ELEMENTS
// =========================================================

console.log("AI Crop Doctor JavaScript loaded.");


// =========================================================
// IMAGE SELECTION
// =========================================================

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];


        if (!file) {
            return;
        }


        // Check image type

        if (
            !file.type.startsWith(
                "image/"
            )
        ) {

            alert(
                "Please select a valid image."
            );

            imageInput.value = "";

            return;
        }


        // Check size

        const maxSize =
            10 * 1024 * 1024;


        if (file.size > maxSize) {

            alert(
                "Image size must be less than 10 MB."
            );

            imageInput.value = "";

            return;
        }


        // Create preview

        const imageURL =
            URL.createObjectURL(
                file
            );


        preview.src =
            imageURL;


        previewContainer.classList.add(
            "show"
        );


        uploadBox.style.display =
            "none";


        result.innerHTML = "";

    }
);


// =========================================================
// REMOVE IMAGE
// =========================================================

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


// =========================================================
// FORM SUBMIT
// =========================================================

form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        // -------------------------------------------------
        // GET FILE
        // -------------------------------------------------

        const file =
            imageInput.files[0];


        if (!file) {

            showError(
                "Please upload a leaf image first."
            );

            return;
        }


        // -------------------------------------------------
        // LOADING
        // -------------------------------------------------

        result.innerHTML = `

            <div class="loading-card">

                <div class="spinner"></div>

                <h3>
                    AI Crop Doctor is analyzing...
                </h3>

                <p>
                    🧠 CNN is detecting the disease.
                </p>

                <p>
                    🤖 Gemini is preparing treatment
                    and prevention advice.
                </p>

                <small>
                    Please wait...
                </small>

            </div>

        `;


        predictButton.disabled =
            true;


        predictButton.innerHTML =
            "⏳ Analyzing...";


        // -------------------------------------------------
        // FORM DATA
        // -------------------------------------------------

        const formData =
            new FormData();


        formData.append(
            "image",
            file
        );


        // -------------------------------------------------
        // SEND TO FLASK
        // -------------------------------------------------

        try {

            console.log(
                "Sending image to:",
                API_URL
            );


            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            // -------------------------------------------------
            // READ RESPONSE
            // -------------------------------------------------

            const data =
                await response.json();


            console.log(
                "Flask response:",
                data
            );


            // -------------------------------------------------
            // ERROR
            // -------------------------------------------------

            if (
                !response.ok ||
                data.status !== "success"
            ) {

                throw new Error(
                    data.message ||
                    "Prediction failed."
                );

            }


            // -------------------------------------------------
            // DISEASE
            // -------------------------------------------------

            const disease =
                data.disease ||
                data.prediction ||
                "Unknown Disease";


            // -------------------------------------------------
            // CONFIDENCE
            // -------------------------------------------------

            let confidence =
                Number(
                    data.confidence
                );


            if (
                Number.isNaN(
                    confidence
                )
            ) {

                confidence = 0;

            }


            confidence =
                confidence.toFixed(2);


            // -------------------------------------------------
            // TOP PREDICTIONS
            // -------------------------------------------------

            const topPredictionsHTML =
                createTopPredictions(
                    data.top_predictions
                );


            // -------------------------------------------------
            // GEMINI
            // -------------------------------------------------

            const treatment =
                formatAdvice(
                    data.treatment
                );


            const prevention =
                formatAdvice(
                    data.prevention
                );


            const cause =
                formatText(
                    data.cause
                );


            const symptoms =
                formatAdvice(
                    data.symptoms
                );


            // -------------------------------------------------
            // GEMINI ERROR
            // -------------------------------------------------

            let geminiWarning = "";


            if (
                data.gemini_error
            ) {

                geminiWarning = `

                    <div class="gemini-warning">

                        ⚠️ Gemini advice could not be generated.

                        <br>

                        ${escapeHTML(
                            data.gemini_error
                        )}

                    </div>

                `;

            }


            // -------------------------------------------------
            // RESULT HTML
            // -------------------------------------------------

            result.innerHTML = `

                <div class="result-header">

                    <div>

                        <span class="result-label">
                            AI ANALYSIS
                        </span>

                        <h2>
                            🌱 Prediction Result
                        </h2>

                    </div>

                    <div class="success-badge">
                        ✓ Complete
                    </div>

                </div>


                <!-- PREDICTION -->

                <div class="prediction-card">

                    <div class="result-row">

                        <span>
                            Disease
                        </span>

                        <strong>
                            ${escapeHTML(
                                disease
                            )}
                        </strong>

                    </div>


                    <div class="result-row">

                        <span>
                            Confidence
                        </span>

                        <strong>
                            ${confidence}%
                        </strong>

                    </div>


                    <div class="confidence-bar">

                        <div
                            style="width:${Math.min(
                                Number(confidence),
                                100
                            )}%"
                        ></div>

                    </div>

                </div>


                <!-- TOP PREDICTIONS -->

                ${topPredictionsHTML}


                <!-- GEMINI INFORMATION -->

                <div class="gemini-title">

                    <span>
                        🤖
                    </span>

                    Gemini AI Agricultural Guidance

                </div>


                <div class="info-grid">


                    <!-- CAUSE -->

                    <div class="info-card">

                        <div class="info-icon">
                            🔬
                        </div>

                        <h3>
                            Cause
                        </h3>

                        <div class="advice-text">
                            ${cause}
                        </div>

                    </div>


                    <!-- SYMPTOMS -->

                    <div class="info-card">

                        <div class="info-icon">
                            🩺
                        </div>

                        <h3>
                            Symptoms
                        </h3>

                        <div class="advice-text">
                            ${symptoms}
                        </div>

                    </div>


                    <!-- TREATMENT -->

                    <div class="info-card treatment">

                        <div class="info-icon">
                            💊
                        </div>

                        <h3>
                            Treatment
                        </h3>

                        <div class="advice-text">
                            ${treatment}
                        </div>

                    </div>


                    <!-- PREVENTION -->

                    <div class="info-card prevention">

                        <div class="info-icon">
                            🛡️
                        </div>

                        <h3>
                            Prevention
                        </h3>

                        <div class="advice-text">
                            ${prevention}
                        </div>

                    </div>


                </div>


                ${geminiWarning}


                <!-- NOTICE -->

                <div class="notice">

                    ⚠️ <strong>Important:</strong>

                    AI-generated agricultural guidance.

                    For pesticide or fungicide use,
                    always follow the product label
                    and local agricultural recommendations.

                </div>

            `;


        }

        catch (error) {

            console.error(
                "Prediction error:",
                error
            );


            showError(
                error.message
            );

        }


        finally {

            predictButton.disabled =
                false;


            predictButton.innerHTML = `

                <span>
                    🔍
                </span>

                Analyze Leaf

                <span>
                    →
                </span>

            `;

        }

    }
);


// =========================================================
// FORMAT ADVICE
// =========================================================

function formatAdvice(text) {

    if (
        !text ||
        String(text).trim() === ""
    ) {

        return `
            <p class="no-data">
                No advice available.
            </p>
        `;

    }


    const lines =
        String(text)
            .split("\n")
            .map(
                line => line.trim()
            )
            .filter(
                line => line.length > 0
            );


    const items = [];


    lines.forEach(
        function (line) {

            line =
                line.replace(
                    /^[-•*]\s*/,
                    ""
                );


            if (
                line.endsWith(":")
            ) {

                return;

            }


            items.push(
                line
            );

        }
    );


    if (
        items.length === 0
    ) {

        return `
            <p>
                ${escapeHTML(text)}
            </p>
        `;

    }


    return `

        <ul>

            ${items.map(
                item => `
                    <li>
                        ${escapeHTML(item)}
                    </li>
                `
            ).join("")}

        </ul>

    `;

}


// =========================================================
// FORMAT NORMAL TEXT
// =========================================================

function formatText(text) {

    if (
        !text ||
        String(text).trim() === ""
    ) {

        return `
            <p class="no-data">
                No information available.
            </p>
        `;

    }


    return `

        <p>
            ${escapeHTML(text)}
        </p>

    `;

}


// =========================================================
// TOP 3 PREDICTIONS
// =========================================================

function createTopPredictions(
    predictions
) {

    if (
        !Array.isArray(
            predictions
        ) ||
        predictions.length === 0
    ) {

        return "";

    }


    return `

        <div class="top-predictions">

            <h3>
                🔎 Other Predictions
            </h3>

            <div class="prediction-list">

                ${predictions.map(
                    function (
                        item,
                        index
                    ) {

                        const disease =
                            item.disease ||
                            "Unknown";


                        const confidence =
                            Number(
                                item.confidence || 0
                            );


                        return `

                            <div class="top-item">

                                <span class="rank">
                                    ${index + 1}
                                </span>

                                <span class="top-name">
                                    ${escapeHTML(
                                        disease
                                    )}
                                </span>

                                <strong>
                                    ${confidence.toFixed(2)}%
                                </strong>

                            </div>

                        `;

                    }
                ).join("")}

            </div>

        </div>

    `;

}


// =========================================================
// ERROR DISPLAY
// =========================================================

function showError(
    message
) {

    result.innerHTML = `

        <div class="error">

            <h3>
                ❌ Prediction Failed
            </h3>

            <p>
                ${escapeHTML(
                    message ||
                    "Something went wrong."
                )}
            </p>

            <small>
                Make sure Flask is running on
                http://127.0.0.1:5000
            </small>

        </div>

    `;

}


// =========================================================
// ESCAPE HTML
// =========================================================

function escapeHTML(
    value
) {

    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );

}