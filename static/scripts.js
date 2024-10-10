document.addEventListener('DOMContentLoaded', () => {
    const cvDropZone = document.getElementById('cvDropZone');
    const cvFile = document.getElementById('cvFile');
    const cvPreview = document.getElementById('cvPreview');
    const offerDropZone = document.getElementById('offerDropZone');
    const offerFile = document.getElementById('offerFile');
    const offerPreview = document.getElementById('offerPreview');
    const matchButton = document.getElementById('matchButton');
    const resultsSection = document.getElementById('results-section');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const loadingPercentage = document.getElementById('loadingPercentage');
    const handshakeAnimation = document.querySelector('.handshake-animation');

    let uploadedCVId = null;
    let uploadedOfferId = null;

    setupDropZone(cvDropZone, cvFile, handleCVUpload);
    setupDropZone(offerDropZone, offerFile, handleOfferUpload);
    matchButton.addEventListener('click', matchCVWithOffer);

    function setupDropZone(dropZone, fileInput, handleFunction) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            dropZone.classList.add('highlight');
        }

        function unhighlight(e) {
            dropZone.classList.remove('highlight');
        }

        dropZone.addEventListener('drop', handleDrop, false);
        dropZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleChange);

        function handleDrop(e) {
            let dt = e.dataTransfer;
            let files = dt.files;
            handleFiles(files);
        }

        function handleChange(e) {
            let files = e.target.files;
            handleFiles(files);
        }

        function handleFiles(files) {
            if (files.length > 0) {
                handleFunction(files[0]);
            }
        }
    }

    async function handleCVUpload(file) {
        if (file.type === 'application/pdf') {
            try {
                await displayPDF(file, cvPreview);
                const response = await uploadFile(file, '/upload_cv/');
                uploadedCVId = response.id;
                cvDropZone.classList.add('file-uploaded');
                showToast('CV téléchargé avec succès!', 'success');
            } catch (error) {
                console.error('Erreur lors du téléchargement du CV:', error);
                showToast('Erreur lors du téléchargement du CV.', 'error');
            }
        } else {
            showToast('Veuillez sélectionner un fichier PDF pour le CV.', 'error');
        }
    }

    async function handleOfferUpload(file) {
        if (file.type === 'application/pdf') {
            try {
                await displayPDF(file, offerPreview);
                const response = await uploadFile(file, '/offres_emploi/');
                uploadedOfferId = response.id;
                offerDropZone.classList.add('file-uploaded');
                showToast('Offre d\'emploi téléchargée avec succès!', 'success');
            } catch (error) {
                console.error('Erreur lors du téléchargement de l\'offre:', error);
                showToast('Erreur lors du téléchargement de l\'offre d\'emploi.', 'error');
            }
        } else {
            showToast('Veuillez sélectionner un fichier PDF pour l\'offre d\'emploi.', 'error');
        }
    }

    function displayPDF(file, previewElement) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = function(e) {
                const typedarray = new Uint8Array(e.target.result);

                pdfjsLib.getDocument(typedarray).promise.then(pdf => {
                    pdf.getPage(1).then(page => {
                        const scale = 1.5;
                        const viewport = page.getViewport({ scale: scale });
                        const canvas = document.createElement('canvas');
                        const context = canvas.getContext('2d');
                        canvas.height = viewport.height;
                        canvas.width = viewport.width;

                        const renderContext = {
                            canvasContext: context,
                            viewport: viewport
                        };
                        page.render(renderContext);

                        previewElement.innerHTML = '';
                        previewElement.appendChild(canvas);
                        resolve();
                    }).catch(reject);
                }).catch(reject);
            };
            reader.onerror = reject;
            reader.readAsArrayBuffer(file);
        });
    }

    async function uploadFile(file, endpoint) {
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await axios.post(endpoint, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            console.log('File uploaded successfully:', response.data);
            return response.data;
        } catch (error) {
            console.error('Error uploading file:', error);
            throw error;
        }
    }

    async function matchCVWithOffer() {
        if (!uploadedCVId || !uploadedOfferId) {
            showToast('Veuillez d\'abord télécharger un CV et une offre d\'emploi.', 'error');
            return;
        }
    
        loadingIndicator.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        matchButton.classList.add('clicked');
        handshakeAnimation.classList.add('shaking');
        
        try {
            const [matchResult] = await Promise.all([
                axios.post('/match_cv_job/', {
                    cv_id: uploadedCVId,
                    job_id: uploadedOfferId
                }),
                simulateLoading()
            ]);
            
            // Arrêter l'animation de serrage de mains
            handshakeAnimation.classList.remove('shaking');
            
            // Animer la séparation des mains
            const leftHand = document.querySelector('.left-hand');
            const rightHand = document.querySelector('.right-hand');
            leftHand.style.transform = 'translateX(-100%) rotate(0deg)';
            rightHand.style.transform = 'translateX(100%) rotate(0deg) scaleX(-1)';
            
            // Attendre la fin de l'animation de séparation
            await new Promise(resolve => setTimeout(resolve, 500));
            
            displayMatchResult(matchResult.data);
        } catch (error) {
            console.error('Erreur lors du matching:', error);
            showToast('Erreur lors du matching. Veuillez réessayer.', 'error');
        } finally {
            loadingIndicator.classList.add('hidden');
            matchButton.classList.remove('clicked');
            
            // Réinitialiser la position des mains pour la prochaine utilisation
            const leftHand = document.querySelector('.left-hand');
            const rightHand = document.querySelector('.right-hand');
            leftHand.style.transform = '';
            rightHand.style.transform = '';
        }
    }

    async function simulateLoading() {
        loadingPercentage.textContent = '0%';
        for (let i = 0; i <= 100; i++) {
            loadingPercentage.textContent = `${i}%`;
            await new Promise(resolve => setTimeout(resolve, 30));
        }
    }

    function displayMatchResult(result) {
        const scorePercentage = document.querySelector('.score-percentage');
        const correspondancesList = document.getElementById('correspondances-list');
        const nonCorrespondancesList = document.getElementById('non-correspondances-list');
        const conclusionText = document.getElementById('conclusion-text');

        // Clear previous results
        correspondancesList.innerHTML = '';
        nonCorrespondancesList.innerHTML = '';

        // Animate score percentage
        animateValue(scorePercentage, 0, result.score.toFixed(2), 2000);

        // Display correspondances with animation
        result.correspondances.forEach((item, index) => {
            const li = document.createElement('li');
            li.textContent = item;
            li.style.opacity = '0';
            li.style.transform = 'translateX(-20px)';
            correspondancesList.appendChild(li);
            setTimeout(() => {
                li.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                li.style.opacity = '1';
                li.style.transform = 'translateX(0)';
            }, index * 100);
        });

        // Display non-correspondances with animation
        result.non_correspondances.forEach((item, index) => {
            const li = document.createElement('li');
            li.textContent = item;
            li.style.opacity = '0';
            li.style.transform = 'translateX(20px)';
            nonCorrespondancesList.appendChild(li);
            setTimeout(() => {
                li.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                li.style.opacity = '1';
                li.style.transform = 'translateX(0)';
            }, index * 100);
        });

        // Display conclusion with typing effect
        const conclusion = result.justification;
        let i = 0;
        conclusionText.textContent = '';
        const typingEffect = setInterval(() => {
            if (i < conclusion.length) {
                conclusionText.textContent += conclusion.charAt(i);
                i++;
            } else {
                clearInterval(typingEffect);
                conclusionText.classList.add('pulse');
            }
        }, 20);

        // Show results section with animation
        resultsSection.classList.remove('hidden');
        setTimeout(() => {
            resultsSection.classList.add('visible');
        }, 100);

        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.classList.add('pulse');
            }
        };
        window.requestAnimationFrame(step);
    }

    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => {
                    document.body.removeChild(toast);
                }, 300);
            }, 3000);
        }, 100);
    }
});