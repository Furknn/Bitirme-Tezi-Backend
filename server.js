const express = require('express');
const multer = require('multer');
const path = require('path');
const MongoClient = require('mongodb').MongoClient;
const dbUrl = "mongodb://localhost:27017/";
const request=require('request')

//Depolama Sekli Konfigurasyonu
const storage = multer.diskStorage({
  destination: './data/uploads/',
  filename: function(req, file, cb){
    cb(null,file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  }
});
//Multer Konfigurasyonu
const upload = multer({
  storage: storage,
  limits:{fileSize: 419000000},
  fileFilter: function(req, file, cb){
    checkFileType(file, cb);
  }
}).single("uploadImage");
function checkFileType(file, cb){
  //İzin verilen dosya tipleri
  const filetypes = /jpeg|jpg|png|gif/;
  //Resim olup olmadığını kontrol et
  const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
  const mimetype = filetypes.test(file.mimetype);
  if(mimetype && extname){
    return cb(null,true);
  } else {
    console.log('Hata:Sadece Resim Yüklenebilir!');
  }
}
//Express app başlat
const app = express();
//Kullanılıcak Klasör
app.use(express.static('./data/detectFace'))
app.get('/', (req, res) => {
    console.log(req.hostname+' Baglandı!')
})
app.post('/upload', (req, res) => {
  upload(req, res, (err) => {
    if(err){
      console.log(err)
    } else {
      if(req.file == undefined){
        console.log("resim secilmedi")
      } else {
        console.log(req.file)
        console.log(req.body)
        request.post('http://localhost:8000',{body:req.file.path},(err,response,body)=>{
        console.log(err)
	      //console.log(response)
        //console.log(body)
        body=body.replace(/'/g, '"');
        //console.log(body)
        bodyJsn=JSON.parse(body)
        bodyJsn["_id"]=req.body.imgId
        bodyJsn["_userId"]=req.body.userId
        
        console.log(bodyJsn)
        res.json(bodyJsn)  
        bodyJsn["_id"]=null//TODO:Remove when you are done testing
        MongoClient.connect(dbUrl, function(err, db) {
          if (err) throw err;
          var dbo = db.db("galleryDb");
          var myobj = bodyJsn;
          try {
            dbo.collection("Images").insertOne(myobj, function(err, res) {
              if (err) throw err;
              console.log("1 adet kayıt yapıldı");
              db.close();
            });
            } catch (error) {
              console.log(error)
            }
         });
       })
      }
    }
  })
})

const port = 3000;

app.listen(port, () => console.log(`Server started on port ${port}`))


