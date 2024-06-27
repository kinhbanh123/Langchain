<script>
    import FormSender from "./lib/FormSender.svelte";
    
    let messages = ["a", "b", "a"]

</script>

<style>
  .box {
    margin: 15px;
  }
</style>

<main>
  <div class = "box">
    {#each messages as message}
        <p>{message}</p>
        <hr>
    {/each}
  </div>
</main>
<FormSender></FormSender>
