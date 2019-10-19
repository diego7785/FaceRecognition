package main

import (
	"context"
	"fmt"
	"time"
	//"strings"
	//"os/exec"

	"github.com/davecgh/go-spew/spew"

	"github.com/aws/aws-lambda-go/events"
	"github.com/gofrs/uuid"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rekognition"
	"github.com/aws/aws-sdk-go/service/rekognition/rekognitioniface"
)

var (
	_rekognitionService rekognitioniface.RekognitionAPI
	collectionID        = "test-video-angelica"
	snsTopicArn         = "arn:aws:sns:us-east-1:530395375560:pruebaAmazonRekognition"
	roleARN             = "arn:aws:iam::530395375560:role/pruebaFaceID"
)

// LambdaRequest body of request
type LambdaRequest struct {
	*events.S3Event
	startingTime time.Time
	bucketName   string
	objectKey    string
	err          error
}

func (req *LambdaRequest) startFaceID(ctx context.Context) error {
	input := &rekognition.StartFaceSearchInput{
		ClientRequestToken: aws.String(uuid.Must(uuid.NewV4()).String()),
		CollectionId:       aws.String(collectionID),
		Video: &rekognition.Video{
			S3Object: &rekognition.S3Object{
				Bucket: aws.String(req.bucketName),
				Name:   aws.String(req.objectKey),
			},
		},
		NotificationChannel: &rekognition.NotificationChannel{
			SNSTopicArn: aws.String(snsTopicArn),
			RoleArn:     aws.String(roleARN),
		},
	}

	out, err := _rekognitionService.StartFaceSearch(input)
	if err != nil {
		fmt.Println("Ops: " + err.Error())

		return err
	}

	spew.Dump(out)
	/*
	salida := spew.Sdump(out)
	jid := strings.SplitAfter(salida, "JobId: \"")
	jid1 := jid[1]
	jid2 := strings.Replace(jid1, "\"\n})\n", "", 1)
	fmt.Println(jid2)
	out, err := exec.Command("sh","-c", "pwd").Output()
	if err != nil{
		fmt.Println("An error ocurred %s", err)
	}
	fmt.Println(out)
	*/
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
		objectKey:  "VideoDiego/video.mp4",
	}

	request.startFaceID(context.Background())
	fmt.Println(request)
	elapsedTime := time.Since(start)
	fmt.Println("Elapsed time: ", elapsedTime)
}
