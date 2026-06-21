let btn = document.getElementById("submit-btn");

btn.addEventListener("click", async () => {
    const file = document.getElementById("input-image").files[0];
    let output = document.getElementById("output-class");
    if(!file)
    {
       
       output.textContent = "GIVE AN INPUT";
       output.hidden = false;
    }
    else {
        const formData = new FormData();
        
        formData.append('file', file);
        const headers = {method: "POST", body: formData};
        try {
        const response = await fetch("/process", headers);
        const json = await response.json();
        output.textContent = json.maximum_class;
        output.hidden = false;
        }
        catch (error)
        {
            console.error(error.message);
        }
        

    }
});