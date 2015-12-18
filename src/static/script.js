$(document).ready(function(){
    $("#submit").click(function(event){
        var uInput = $("#user-input").val();
        $.ajax({
              type: "POST",
              url: '/learning',
              data: JSON.stringify({userInput: uInput}),
              contentType: 'application/json',
              success: function(response){
                   $("#result1").text(response.anger);
                    $("#result2").text(response.disgust);
                    $("#result3").text(response.happy);
                    $("#result4").text(response.sad);
                    $("#result5").text(response.surprise);
                },
          });
    });
});