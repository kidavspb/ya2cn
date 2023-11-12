document.getElementById('submitBtn').addEventListener('click', submitComment);

document.getElementById('commentInput').addEventListener('keydown', function(event) {
    if (event.key === 'Enter' || event.keyCode === 13) {
        submitComment();
    }
});

function submitComment() {
    const commentInput = document.getElementById('commentInput');
    const commentsHistory = document.getElementById('commentsHistory');

    if (commentInput.value.trim() !== "") {
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                comment: commentInput.value.trim()
            })
        })
        .then(response => response.json())
        .then(data => {
            const commentDiv = document.createElement('div');
            commentDiv.className = 'comment';
            commentDiv.textContent = (data.result[0] == "Positive" ? "😃" : "😔") + "("+ Math.round(data.result[1]*10)/10 +")" + " " + commentInput.value.trim();
            commentsHistory.prepend(commentDiv);
            commentInput.value = "";

            // Обновляем полоску после добавления комментария
            updateSentimentBar();
        });
    } else {
        alert("Please write a comment before submitting.");
    }
}

function updateSentimentBar() {
    fetch('/get_sentiment_ratio')
    .then(response => response.json())
    .then(data => {
        const positiveBar = document.getElementById('positiveBar');
        const negativeBar = document.getElementById('negativeBar');

        positiveBar.style.width = data.positive_percentage + '%';
        negativeBar.style.width = data.negative_percentage + '%';
    })
    .catch(error => console.error('Error:', error));
}


// Вызываем функцию сброса счетчиков при загрузке страницы
window.addEventListener('load', resetCountersOnServer);

function resetCountersOnServer() {
    fetch('/reset_counters', {
        method: 'POST'
    })
    .catch(error => console.error('Error:', error));
}

