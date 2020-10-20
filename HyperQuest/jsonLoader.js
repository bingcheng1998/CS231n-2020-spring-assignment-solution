'use strict';

const fs = require('fs');

var hs = 3, bs = 10, lr = 1e-3, reg = 1, num_epoch = 8

var filename = 'json/' + hs + '-' + bs + '-' + lr + '-' + reg + '-' + num_epoch + '.json'
console.log('filename = '+filename)
let rawdata = fs.readFileSync(filename);
let student = JSON.parse(rawdata);
console.log(student['W1'].length);
// console.log(student['W1'][0].length);
// console.log(student['W1'][0][0].length);