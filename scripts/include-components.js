document.addEventListener("DOMContentLoaded", function() {
  // Load components marked with data-include attribute
  document.querySelectorAll('[data-include]').forEach(element => {
    const filePath = element.getAttribute('data-include');
    fetch(filePath)
      .then(response => response.text())
      .then(data => {
        element.innerHTML = data;
        
        // Re-initialize Bootstrap components if needed
        if (element.querySelector('.navbar')) {
          // Add navbar toggle functionality
          const navLinks = element.querySelectorAll("#navbarNav a");
          navLinks.forEach(link => {
            link.addEventListener("click", function() {
              const navbarToggler = document.querySelector(".navbar-toggler");
              if (navbarToggler.getAttribute("aria-expanded") === "true") {
                navbarToggler.click();
              }
            });
          });
        }
      });
  });
}); 