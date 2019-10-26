//Creating a collection with GO

package main

import (
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/rekognition"
)

var collectionName = "test-video-diego"
func main(){
	start := time.Now()
	time.Sleep(time.Second*2)
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	)

	if err != nil {
		fmt.Println("Ops: " + err.Error())
		return
	}

	_rekognitionService := rekognition.New(sess)
	input := &rekognition.CreateCollectionInput{
		CollectionId: aws.String(collectionName),
	}

	result, err := _rekognitionService.CreateCollection(input)
	if err != nil {
		fmt.Println("Ops: ", err.Error())
		return
	}

	fmt.Println(result)
	elapsedTime := time.Since(start)
	fmt.Println("Elapsed time: ", elapsedTime)
}