function validateForm() {
    const inputs = [
        'Zipcode',
        'User_Frequency_Per_Day',
        'Time_Difference_Hours',
        'Merchant_Category_Code',
        'Transaction_Amount'
    ];
    
    for (let name of inputs) {
        let value = document.forms["fraud-form"][name].value;
        if (!value || isNaN(parseFloat(value))) {
            alert(`Please enter a valid number for ${name}`);
            return false;
        }
    }
    
    let state = document.forms["fraud-form"]["Merchant_State_Code"].value;
    let city = document.forms["fraud-form"]["Merchant_City_Code"].value;
    if (!state || state === "") {
        alert("Please select a valid Merchant State");
        return false;
    }
    if (!city || city === "") {
        alert("Please select a valid Merchant City");
        return false;
    }
    
    return true;
}