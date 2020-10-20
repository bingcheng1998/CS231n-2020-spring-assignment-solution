function loadJSON(callback) {

    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', '../json/3-10-0.001-1-8.json', true);
    xobj.onreadystatechange = function() {
        if (xobj.readyState == 4 && xobj.status == "200") {
            callback(xobj.responseText);
        }
    }
    xobj.send(null);
}

var plot = function (student, i, canvasId) {
	var rgbdata = student['W1'][i];
	var c = document.getElementById(canvasId); 
	var ctx = c.getContext("2d"); 

	var r,g,b; 

	for(var i=0; i< rgbdata.length; i++){ 
		for(var j=0; j< rgbdata[0].length; j++){ 
			r = rgbdata[i][j][0]; 
			g = rgbdata[i][j][1];	 
			b = rgbdata[i][j][2];		 
			ctx.fillStyle = "rgba("+r+","+g+","+b+", 1)";  
			ctx.fillRect( j, i, 1, 1 ); 
		} 
	} 
}


loadJSON(function(response) {
    let student = JSON.parse(response);
	var rgbdata = student['W1'][0];
	console.log(rgbdata);
	plot(student, 0, 'myCanvas0')
	plot(student, 1, 'myCanvas1')
	plot(student, 2, 'myCanvas2')
});

