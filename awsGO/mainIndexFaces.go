package main

import (
	"fmt"
	"time"

	"github.com/davecgh/go-spew/spew"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rekognition"
	"github.com/aws/aws-sdk-go/service/rekognition/rekognitioniface"
)

var (
	_rekognitionService rekognitioniface.RekognitionAPI
	collectionID        = "test-video-angelica"
)

// LambdaRequest body of request
type LambdaRequest struct {
	*events.S3Event
	bucketName string
	objectKey  string
	err        error
}

func (req *LambdaRequest) addFaces() error {
	input := &rekognition.IndexFacesInput{
		ExternalImageId:     aws.String("test-video-1.png"),
		CollectionId:        aws.String(collectionID),
		DetectionAttributes: []*string{aws.String("ALL")},
		Image: &rekognition.Image{
			S3Object: &rekognition.S3Object{
				Bucket: aws.String(req.bucketName),
				Name:   aws.String(req.objectKey),
			},
		},
	}

	spew.Dump(input)

	out, err := _rekognitionService.IndexFaces(input)
	if err != nil {
		return err
	}

	spew.Dump(out)

	return nil
}

func main() {
	start := time.Now()
	time.Sleep(time.Second*2)
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	)

	if err != nil {
		fmt.Println("Ops: " + err.Error())
		return
	}

	_rekognitionService = rekognition.New(sess)

	request := LambdaRequest{
		bucketName: "prueba-face-recognition",
		objectKey:  "VideoAngelica/Angelica.png",
	}

	err = request.addFaces()
	if err != nil {
		fmt.Println("Ops: " + err.Error())
		return
	}
	elapsedTime := time.Since(start)
	fmt.Println("Elapsed time: ", elapsedTime)
}
