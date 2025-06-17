const canvas = document.getElementById("canvas");
canvas.style.position = "absolute";
canvas.style.top = "150px";
canvas.style.left = "50px";

canvas.height = window.innerHeight - 200;
canvas.width = window.innerWidth -100;

let context = canvas.getContext("2d");
let start_background_color = "white"

context.fillStyle = start_background_color;
context.fillRect(0, 0, canvas.width, canvas.height);

let draw_color = "black";
let draw_width = "2";
let is_drawing = false;


array_curves = [];
let index = -1;



function change_color(element){
    draw_color = element.style.background;
}


canvas.addEventListener("touchstart", start, false);
canvas.addEventListener("touchmove", draw, false);
//canvas.addEventListener("mousedown", start, false);
canvas.addEventListener("mousemove", draw, false);

canvas.addEventListener("touchend", stop, false);
canvas.addEventListener("mouseup", stop, false);
canvas.addEventListener("mouseout", stop, false);

canvas.addEventListener('mousedown', function(event) {
    is_drawing = true;
    lastX = event.clientX - canvas.offsetLeft;
    lastY = event.clientY - canvas.offsetTop;
});


function start(event) {
    is_drawing = true;
    context.beginPath();
    context.moveTo(event.clientX - canvas.offsetLeft,
        event.clientY - canvas.offsetTop);
    event.preventDefault();
}





function draw(event) {
    if (is_drawing) {
        //context.lineTo(event.clientX - canvas.offsetLeft,
        //    event.clientY - canvas.offsetTop);
        currentX = event.clientX - canvas.offsetLeft;
        currentY = event.clientY - canvas.offsetTop;
        
        context.strokeStyle = draw_color;
        context.lineWidth = draw_width;
        context.lineJoin = "round";
        context.lineCap = "round";

        context.beginPath();
        context.moveTo(lastX, lastY);
        context.quadraticCurveTo(lastX, lastY, currentX, currentY);
        context.stroke();
        context.closePath();

        lastX = currentX;
        lastY = currentY;
        context.stroke();

        event.preventDefault();
    }
else
    return;
}



function stop(event) {
    if (is_drawing) {
        context.stroke();
        context.closePath();
        is_drawing = false;
    }
    event.preventDefault();

    if (event.type != "mouseout"){
        array_curves.push(context.getImageData(0, 0, canvas.width, canvas.height));
        index += 1;
    }
    console.log(array_curves)
}


function clear_canvas() {
    context.fillStyle = start_background_color;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillRect(0, 0, canvas.width, canvas.height);

    array_curves = [];
    index = -1;
}


function drawFlowerPattern() {
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var centerX = canvas.width / 2;
    var centerY = canvas.height / 2;

    ctx.save();
    ctx.translate(centerX, centerY);

    ctx.putImageData(imageData, 0, 0);

    for (var i = 1; i < 6; i++) {
        ctx.save();
        ctx.rotate(i * 30 * Math.PI / 180);
        ctx.translate(-centerX, -centerY);
        ctx.putImageData(imageData, 0, 0);
        ctx.translate(centerX, centerY);
    }

    ctx.restore();
}


function createSymmetricImage() {
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext('2d');

    // Get the image data for the original image
    const originalData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Create a new image data object for the symmetric image
    const symmetricData = ctx.createImageData(canvas.width, canvas.height);

    // Copy the original image to the symmetric image
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const sx = canvas.width - x - 1;
            const sy = y;
            for (let i = 0; i < 4; i++) {
                symmetricData.data[(y * canvas.width + x) * 4 + i] = originalData.data[(sy * canvas.width + sx) * 4 + i];
            }
        }
    }

    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            for (let i = 0; i < 4; i++) {
                let originalValue = originalData.data[(y * canvas.width + x) * 4 + i];
                let symmetricValue = symmetricData.data[(y * canvas.width + x) * 4 + i];
                let combinedValue = Math.min(originalValue, symmetricValue);
                symmetricData.data[(y * canvas.width + x) * 4 + i] = combinedValue; // Ensure value doesn't exceed 255
            }
        }
    }
   
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(symmetricData, 0, 0);
}



function rotateImageBy30Degrees() {
    var canvas = document.getElementById("canvas"); 
    var ctx = canvas.getContext('2d'); // Create a temporary canvas to draw the rotated image 
    var tempCanvas = document.createElement("canvas"); 
    var tempCtx = tempCanvas.getContext('2d'); 
    tempCanvas.width = canvas.width; tempCanvas.height = canvas.height; // Get the image data for the original image 
    var image = new Image(); 
    image.src = canvas.toDataURL(); 
    image.onload = function() { // Rotate the temporary canvas by 30 degrees 
        tempCtx.translate(tempCanvas.width / 2, tempCanvas.height / 2); 
        tempCtx.rotate(30 * Math.PI / 180); 
        tempCtx.translate(-tempCanvas.width / 2, -tempCanvas.height / 2); // Draw the original image onto the rotated canvas 
        tempCtx.drawImage(image, 0, 0); // Clear the original canvas and draw the rotated image 
        ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(tempCanvas, 0, 0); } 
    }


    function overlayRotatedImage() {
        var canvas = document.getElementById("canvas");
        var ctx = canvas.getContext('2d');
    
        // Create a temporary canvas to draw the rotated image
        var tempCanvas = document.createElement("canvas");
        var tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
    
        // Get the image data for the original image
        var image = new Image();
        image.src = canvas.toDataURL();
        image.onload = function() {
            // Draw the original image on the original canvas
            ctx.drawImage(image, 0, 0);
    
            // Rotate the temporary canvas by 30 degrees
            tempCtx.translate(tempCanvas.width / 2, tempCanvas.height / 2);
            tempCtx.rotate(30 * Math.PI / 180);
            tempCtx.translate(-tempCanvas.width / 2, -tempCanvas.height / 2);
    
            // Draw the original image onto the rotated canvas
            tempCtx.drawImage(image, 0, 0);
    
            // Set transparency for the overlay
            ctx.globalAlpha = 0.1; // Adjust the value (between 0 and 1) to control transparency
    
            // Draw the rotated image on top of the original image
            ctx.drawImage(tempCanvas, 0, 0);
        }
    }
    