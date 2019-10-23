//Uploading files to s3

package main

import(
	"fmt"
	"bytes"
	"net/http"
	"os"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

const (
	region = "us-east-1"
	bucket = "prueba-face-recognition"
)

func main(){
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String(region)},
	)

	if err != nil{
		fmt.Println("Ops: ", err.Error())
	}

	//Uploading file
	err = AddFileToS3(sess, "diego.jpg") 
	if err != nil{
		fmt.Println("Ops: ", err.Error())
	}
}

//Adding files to S3 require a pre-built aws session 
func AddFileToS3(s *session.Session, fileDir string) error{
	file, err := os.Open(fileDir)
	if err != nil{
		fmt.Println("Ops: ", err.Error())	
	}

	defer file.Close()

	//Get file size and get info into a buffer
	fileInfo, _ := file.Stat()
	var size int64 = fileInfo.Size()
	buffer := make([]byte, size)
	file.Read(buffer)

	//Config settings, setting the bucket, filename, content-type...

	_, err = s3.New(s).PutObject(&s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key: aws.String(fileDir),
		ACL: aws.String("private"),
		Body:                 bytes.NewReader(buffer),
        ContentLength:        aws.Int64(size),
        ContentType:          aws.String(http.DetectContentType(buffer)),
        ContentDisposition:   aws.String("attachment"),
        ServerSideEncryption: aws.String("AES256"),
	})	

	return err
}
