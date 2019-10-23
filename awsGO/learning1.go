//Learning how to use S3 with GO

package main

import (
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/aws"
)

func main(){
	//Initialize the session on s3 in us-east-1 region
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	)

	if err != nil {
		fmt.Println("Ops: ", err.Error())
	} else {

		//Create s3 service client
		svc := s3.New(sess)

		//Listing buckets
		result, err := svc.ListBuckets(nil)
		if err != nil{
			fmt.Println("Unable to list buckets: %v", err.Error())
			os.Exit(1)
		}

		fmt.Println("Buckets: \n")

		for _, b := range result.Buckets{
			fmt.Println(aws.StringValue(b.Name), "\n")

		}
	}

}
	

