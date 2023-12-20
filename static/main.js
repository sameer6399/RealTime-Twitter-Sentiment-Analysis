$(function() { 
    $('#sidebarCollapse').on('click', function() {
      $('#sidebar, #content').toggleClass('active');
    });
  });

  $('#btn-one').click(function() {
    $('#btn-one').html('<span class="spinner-border spinner-border-sm mr-2" role="status" aria-hidden="true"></span>Loading.. Please wait').attr('disabled', true);
  });
  
  
  $('.btn').on('click', function() {
    var $this = $(this);
  $this.button('Loading.. Please wait');
    setTimeout(function() {
       $this.button('reset');
   }, 8000);
});
  